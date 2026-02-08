"""Audio transcription CLI tool.

Transcribes voice recordings to SRT format with speaker diarization using
Gemini 3 models via the OpenRouter API. Produces SRT output files, sidecar
metadata JSON, and prints metadata to stdout for agent consumption.
"""

import argparse
import base64
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage
from pydantic import BaseModel, ConfigDict, ValidationError

# =============================================================================
# Enums
# =============================================================================


class AudioFormat(StrEnum):
    """Supported audio input formats."""

    MP3 = "mp3"
    M4A = "m4a"
    WAV = "wav"
    AAC = "aac"
    OGG = "ogg"
    FLAC = "flac"
    AIFF = "aiff"


class ThinkingEffort(StrEnum):
    """Thinking effort levels for the reasoning model."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelChoice(StrEnum):
    """Available model choices."""

    FLASH = "google/gemini-3-flash-preview"
    PRO = "google/gemini-3-pro-preview"


# =============================================================================
# Pydantic Models (LLM Response Schema)
# =============================================================================


class Segment(BaseModel):
    """A single transcription segment with speaker and timestamps."""

    model_config = ConfigDict(frozen=True)

    speaker: str
    start_time: str
    end_time: str
    text: str


class SpeakerInfo(BaseModel):
    """Speaker identification metadata."""

    model_config = ConfigDict(frozen=True)

    label: str
    gender: str
    name_guess: str


class TranscriptionResponse(BaseModel):
    """Complete LLM transcription response combining segments and metadata."""

    model_config = ConfigDict(frozen=True)

    segments: list[Segment]
    title: str
    summary: str
    language: str
    speakers: list[SpeakerInfo]
    topics: list[str]
    key_terms: list[str]


# =============================================================================
# Pydantic Models (Output Serialization)
# =============================================================================


class UsageCost(BaseModel):
    """API usage and cost information."""

    model_config = ConfigDict(frozen=True)

    total: float | None
    audio_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    prompt_tokens: int | None


class TranscriptionMetadata(BaseModel):
    """Metadata written to .meta.json and stdout."""

    model_config = ConfigDict(frozen=True)

    title: str
    summary: str
    language: str
    speakers: list[SpeakerInfo]
    topics: list[str]
    key_terms: list[str]
    duration_seconds: int
    segment_count: int
    cost: UsageCost
    model: str
    input_file: str
    output_file: str


# =============================================================================
# Internal Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Result of an API transcription call."""

    success: bool
    raw_content: str
    usage: CompletionUsage | None
    error: str | None


# =============================================================================
# TypedDicts for API Message Construction
# =============================================================================


class InputAudioData(TypedDict):
    """Audio data payload for the API."""

    data: str
    format: str


class InputAudioPart(TypedDict):
    """Audio content part in a message."""

    type: Literal["input_audio"]
    input_audio: InputAudioData


class TextPart(TypedDict):
    """Text content part in a message."""

    type: Literal["text"]
    text: str


type MessageContent = list[TextPart | InputAudioPart]


class UserMessage(TypedDict):
    """User message with multimodal content."""

    role: Literal["user"]
    content: MessageContent


class SystemMessage(TypedDict):
    """System message with text content."""

    role: Literal["system"]
    content: str


class ReasoningConfig(TypedDict):
    """Reasoning effort configuration for extra_body."""

    effort: str


class ExtraBody(TypedDict, total=False):
    """Extra body parameters for the API call."""

    reasoning: ReasoningConfig


# =============================================================================
# Constants
# =============================================================================

MAX_FILE_SIZE: Final = 20 * 1024 * 1024
MAX_OUTPUT_TOKENS: Final = 65_536
PROGRESS_INTERVAL_SEC: Final = 10
TOKENS_PER_SECOND: Final = 32


# =============================================================================
# Functions
# =============================================================================


def _get_client() -> OpenAI:
    """Initialize OpenAI client pointed at OpenRouter.

    Loads environment variables from .env files via python-dotenv,
    then reads OPENROUTER_API_KEY.

    :return: Initialized OpenAI client with OpenRouter base URL.
    :raises SystemExit: If OPENROUTER_API_KEY is not set.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None:
        sys.stderr.write("Error: OPENROUTER_API_KEY environment variable not set\n")
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _validate_input(path: Path) -> tuple[Path, AudioFormat]:
    """Validate audio input file exists, has supported format, and is within size limit.

    :param path: Path to the audio file.
    :return: Tuple of resolved path and audio format.
    :raises SystemExit: If validation fails.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        sys.stderr.write(f"Error: File not found: {resolved}\n")
        sys.exit(1)

    extension = resolved.suffix.lstrip(".").lower()
    try:
        audio_format = AudioFormat(extension)
    except ValueError:
        supported = ", ".join(f.value for f in AudioFormat)
        sys.stderr.write(f"Error: Unsupported audio format '{extension}'. Supported: {supported}\n")
        sys.exit(1)

    file_size = resolved.stat().st_size
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE / (1024 * 1024)
        sys.stderr.write(f"Error: File too large ({size_mb:.1f} MB). Maximum: {limit_mb:.0f} MB\n")
        sys.exit(1)

    return resolved, audio_format


def _build_messages(audio_b64: str, audio_format: AudioFormat) -> list[SystemMessage | UserMessage]:
    """Build API messages with system prompt and audio content.

    :param audio_b64: Base64-encoded audio data.
    :param audio_format: Audio format enum value.
    :return: List of system and user messages.
    """
    system_msg = SystemMessage(
        role="system",
        content=(
            "<role>You are a transcription specialist.</role>\n"
            "<task>Accurately transcribe audio with speaker diarization and timestamps.</task>\n"
            "<rules>\n"
            "- Label speakers as speaker_1, speaker_2, etc.\n"
            "- Timestamps in HH:MM:SS format for each segment\n"
            "- Transcribe all spoken content faithfully, including filler words\n"
            "- If a word is unclear, transcribe your best interpretation\n"
            "- Detect the primary language automatically\n"
            "- Provide a brief summary and descriptive title\n"
            "- Identify topics and key terms\n"
            "- Guess speaker names and genders from context if possible\n"
            "</rules>\n"
            "<example>\n"
            '{"segments": [{"speaker": "speaker_1", "start_time": "00:00:00", '
            '"end_time": "00:00:05", "text": "Hello, welcome to the meeting."}], '
            '"title": "Team standup", "summary": "Brief team standup discussion.", '
            '"language": "en", "speakers": [{"label": "speaker_1", "gender": "male", '
            '"name_guess": "John"}], "topics": ["standup"], "key_terms": ["meeting"]}\n'
            "</example>"
        ),
    )

    user_msg = UserMessage(
        role="user",
        content=[
            TextPart(type="text", text="Transcribe the attached audio recording."),
            InputAudioPart(
                type="input_audio",
                input_audio=InputAudioData(data=audio_b64, format=audio_format.value),
            ),
        ],
    )

    return [system_msg, user_msg]


def _build_response_format() -> dict[str, object]:
    """Build the response_format parameter using Pydantic-generated JSON Schema.

    :return: Response format dict for the API call.
    """
    schema = TranscriptionResponse.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "transcription_response",
            "strict": True,
            "schema": schema,
        },
    }


def _stream_transcription(
    client: OpenAI,
    model: ModelChoice,
    messages: list[SystemMessage | UserMessage],
    response_format: dict[str, object],
    thinking_effort: ThinkingEffort,
) -> StreamResult:
    """Stream the transcription API call with progress reporting.

    Reports progress to stderr every ``PROGRESS_INTERVAL_SEC`` seconds.

    :param client: OpenAI client.
    :param model: Model to use.
    :param messages: API messages.
    :param response_format: JSON schema response format.
    :param thinking_effort: Reasoning effort level.
    :return: StreamResult with raw content and usage.
    """
    extra_body = ExtraBody(reasoning=ReasoningConfig(effort=thinking_effort.value))
    content_parts: list[str] = []
    usage: CompletionUsage | None = None
    last_progress = time.monotonic()
    start_time = last_progress

    try:
        stream = client.chat.completions.create(
            model=model.value,
            messages=messages,
            response_format=response_format,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SEC:
                elapsed = int(now - start_time)
                total_chars = sum(len(p) for p in content_parts)
                sys.stderr.write(f"[{elapsed}s] {total_chars:,} chars received\n")
                last_progress = now

    except Exception as e:  # noqa: BLE001  # OpenAI SDK raises various exception types during streaming
        return StreamResult(
            success=False,
            raw_content="".join(content_parts),
            usage=None,
            error=str(e),
        )

    return StreamResult(success=True, raw_content="".join(content_parts), usage=usage, error=None)


def _parse_response(raw_json: str) -> TranscriptionResponse | None:
    """Parse and validate the raw JSON response using Pydantic.

    :param raw_json: Raw JSON string from the API.
    :return: Parsed TranscriptionResponse, or None if validation fails.
    """
    try:
        return TranscriptionResponse.model_validate_json(raw_json)
    except ValidationError:
        return None


def _validate_segments(segments: list[Segment]) -> list[str]:
    """Validate transcription segments for semantic correctness.

    Checks timestamp format, monotonicity, and speaker label presence.
    Returns warning messages (empty list if all valid).

    :param segments: List of transcription segments.
    :return: List of warning messages.
    """
    warnings: list[str] = []
    time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    prev_start = ""

    for i, segment in enumerate(segments):
        if not time_pattern.match(segment.start_time):
            warnings.append(f"Segment {i + 1}: unparseable start_time '{segment.start_time}'")
        if not time_pattern.match(segment.end_time):
            warnings.append(f"Segment {i + 1}: unparseable end_time '{segment.end_time}'")
        if prev_start and segment.start_time < prev_start:
            warnings.append(f"Segment {i + 1}: non-monotonic start_time '{segment.start_time}' < '{prev_start}'")
        if not segment.speaker:
            warnings.append(f"Segment {i + 1}: empty speaker label")
        prev_start = segment.start_time

    return warnings


def _parse_end_time_seconds(end_time: str) -> int:
    """Parse HH:MM:SS timestamp to total seconds.

    :param end_time: Timestamp string in HH:MM:SS format.
    :return: Total seconds, or 0 if unparseable.
    """
    parts = end_time.split(":")
    if len(parts) != 3:  # HH:MM:SS has exactly 3 parts
        return 0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return 0


def _build_metadata(
    response: TranscriptionResponse,
    usage: CompletionUsage | None,
    model: ModelChoice,
    input_file: str,
    output_file: str,
) -> TranscriptionMetadata:
    """Build output metadata from transcription response and API usage.

    :param response: Parsed transcription response.
    :param usage: API usage statistics.
    :param model: Model used for transcription.
    :param input_file: Input filename.
    :param output_file: Output SRT filename.
    :return: TranscriptionMetadata for serialization.
    """
    duration_seconds = 0
    if response.segments:
        duration_seconds = _parse_end_time_seconds(response.segments[-1].end_time)

    audio_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    prompt_tokens: int | None = None
    total_cost: float | None = None

    if usage is not None:
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        if usage.prompt_tokens_details is not None:
            audio_tokens = usage.prompt_tokens_details.audio_tokens
        if usage.completion_tokens_details is not None:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens

    cost = UsageCost(
        total=total_cost,
        audio_tokens=audio_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_tokens=prompt_tokens,
    )

    return TranscriptionMetadata(
        title=response.title,
        summary=response.summary,
        language=response.language,
        speakers=response.speakers,
        topics=response.topics,
        key_terms=response.key_terms,
        duration_seconds=duration_seconds,
        segment_count=len(response.segments),
        cost=cost,
        model=model.value,
        input_file=input_file,
        output_file=output_file,
    )


def _segments_to_srt(segments: list[Segment]) -> str:
    """Convert transcription segments to SRT subtitle format.

    :param segments: List of transcription segments.
    :return: SRT-formatted string.
    """
    lines: list[str] = []
    for i, segment in enumerate(segments, 1):
        lines.extend(
            (
                str(i),
                f"{segment.start_time},000 --> {segment.end_time},000",
                f"[{segment.speaker}] {segment.text}",
                "",
            )
        )
    return "\n".join(lines)


def _save_raw_response(output_dir: Path, stem: str, raw_content: str) -> Path:
    """Save raw LLM response to a .raw.json file.

    :param output_dir: Directory to write to.
    :param stem: Filename stem (without extension).
    :param raw_content: Raw response content.
    :return: Path to the written file.
    """
    raw_path = output_dir / f"{stem}.raw.json"
    raw_path.write_text(raw_content, encoding="utf-8")
    return raw_path


def _run_transcription(args: argparse.Namespace) -> int:
    """Run the transcription pipeline.

    :param args: Parsed command-line arguments.
    :return: Exit code (0 for success, 1 for error).
    """
    # Validate input
    resolved_path, audio_format = _validate_input(Path(args.file))
    input_stem = resolved_path.stem
    output_dir: Path = args.output_dir

    # Read and encode audio
    sys.stderr.write(f"Reading: {resolved_path}\n")
    audio_bytes = resolved_path.read_bytes()
    sys.stderr.write(f"File size: {len(audio_bytes) / (1024 * 1024):.1f} MB\n")

    # Determine model and thinking effort
    model = ModelChoice.PRO if args.pro else ModelChoice.FLASH
    thinking_effort = ThinkingEffort(args.thinking_effort)
    if model == ModelChoice.PRO and thinking_effort == ThinkingEffort.MINIMAL:
        sys.stderr.write("Pro model does not support 'minimal' thinking effort, using 'low'\n")
        thinking_effort = ThinkingEffort.LOW

    # Build request and stream API call
    messages = _build_messages(base64.b64encode(audio_bytes).decode("ascii"), audio_format)
    sys.stderr.write(f"Transcribing with {model.value} (thinking: {thinking_effort.value})...\n")
    result = _stream_transcription(_get_client(), model, messages, _build_response_format(), thinking_effort)

    if not result.success:
        if result.raw_content:
            sys.stderr.write(
                f"Raw response saved to: {_save_raw_response(output_dir, input_stem, result.raw_content)}\n"
            )
        sys.stderr.write(f"Error: {result.error}\n")
        return 1

    # Parse response
    response = _parse_response(result.raw_content)
    if response is None:
        sys.stderr.write(
            f"Error: Schema non-compliance. Raw response saved to: "
            f"{_save_raw_response(output_dir, input_stem, result.raw_content)}\n"
        )
        return 1

    # Validate segments
    for warning in _validate_segments(response.segments):
        sys.stderr.write(f"Warning: {warning}\n")

    # Write SRT file
    srt_filename = f"{input_stem}.srt"
    srt_path = output_dir / srt_filename
    srt_path.write_text(_segments_to_srt(response.segments), encoding="utf-8")
    sys.stderr.write(f"SRT saved to: {srt_path}\n")

    # Build and write metadata
    metadata = _build_metadata(response, result.usage, model, resolved_path.name, srt_filename)
    metadata_json = metadata.model_dump_json(indent=2)

    meta_path = output_dir / f"{input_stem}.meta.json"
    meta_path.write_text(metadata_json, encoding="utf-8")
    sys.stderr.write(f"Metadata saved to: {meta_path}\n")

    # Print metadata to stdout for agent consumption
    print(metadata_json)

    return 0


def main() -> None:
    """Entry point for audio-transcribe command."""
    parser = argparse.ArgumentParser(
        prog="audio-transcribe",
        description="Transcribe audio recordings to SRT format with speaker diarization using Gemini via OpenRouter.",
        epilog="""
OUTPUT FORMAT:
  Standard SRT subtitle file with speaker labels:
    1
    00:00:00,000 --> 00:00:05,000
    [speaker_1] Good morning, let's start with the agenda.

    2
    00:00:05,000 --> 00:00:12,000
    [speaker_2] Sure, I wanted to discuss the migration timeline.

SUPPORTED FORMATS:
  mp3, m4a, wav, aac, ogg, flac, aiff

ENVIRONMENT:
  OPENROUTER_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Basic transcription (Flash model, minimal thinking)
  audio-transcribe recording.mp3

  # Output to specific directory
  audio-transcribe -o ./transcripts/ recording.mp3

  # Use Pro model for higher quality
  audio-transcribe --pro recording.mp3

  # Enable more thinking for complex diarization
  audio-transcribe --thinking-effort low recording.mp3
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        metavar="FILE",
        help="path to audio file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(),
        metavar="DIR",
        help="output directory for SRT and metadata files (default: current directory)",
    )
    parser.add_argument(
        "--pro",
        action="store_true",
        help="use Gemini 3 Pro instead of Flash",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=[e.value for e in ThinkingEffort],
        default=ThinkingEffort.MINIMAL.value,
        metavar="LEVEL",
        help="thinking effort: minimal, low, medium, high (default: minimal)",
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = _run_transcription(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
