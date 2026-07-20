// ── Shared finish_reason / stop_reason mapping ──────────────────────────────
//
// Both directions of this mapping used to be hand-copied in three places:
// `translate_openai_to_anthropic_response`, `translate_anthropic_to_openai_response`
// (utils/translate.rs), and the streaming reducer's private `map_chat_finish_reason`
// (stream/openai/mod.rs). All three needed to agree on the same table, so this
// is the single source of truth all of them now call into. Exported publicly
// since callers embedding this library (e.g. a gateway that builds its own
// Anthropic-shaped SSE/JSON from a worker's OpenAI `finish_reason` without going
// through the full response-translation functions) need the same mapping too.

/// Map an OpenAI-style `finish_reason` to the equivalent Anthropic `stop_reason`.
/// Unrecognized values pass through unchanged rather than being coerced to a
/// default, so callers can tell "we mapped this" from "we didn't recognize this".
pub fn openai_finish_reason_to_anthropic_stop_reason(reason: &str) -> String {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        // Anthropic's stop_reason enum has no direct "this got filtered"
        // value, and `stop_sequence` specifically claims a caller-supplied
        // stop string was matched — which is a different, misleading claim.
        // `end_turn` is the closest non-misleading value: it says "the turn
        // ended" without asserting anything about a stop sequence, and
        // Claude Code doesn't branch on stop_sequence vs. end_turn
        // differently, so this doesn't change loop behavior either way.
        "content_filter" => "end_turn",
        other => other,
    }
    .to_string()
}

/// Map an Anthropic-style `stop_reason` to the equivalent OpenAI `finish_reason`.
/// Inverse of [`openai_finish_reason_to_anthropic_stop_reason`]; same
/// pass-through-if-unrecognized behavior.
pub fn anthropic_stop_reason_to_openai_finish_reason(reason: &str) -> String {
    match reason {
        "end_turn" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        // OpenAI's finish_reason enum has no stop-sequence-specific value —
        // it reports a matched stop string the same way as any other clean
        // stop. `content_filter` would falsely claim moderation flagged the
        // output, which a stop-sequence hit says nothing about.
        "stop_sequence" => "stop",
        other => other,
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_openai_reasons() {
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("stop"), "end_turn");
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("length"), "max_tokens");
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("tool_calls"), "tool_use");
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("function_call"), "tool_use");
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("content_filter"), "end_turn");
    }

    #[test]
    fn maps_known_anthropic_reasons() {
        assert_eq!(anthropic_stop_reason_to_openai_finish_reason("end_turn"), "stop");
        assert_eq!(anthropic_stop_reason_to_openai_finish_reason("max_tokens"), "length");
        assert_eq!(anthropic_stop_reason_to_openai_finish_reason("tool_use"), "tool_calls");
        assert_eq!(anthropic_stop_reason_to_openai_finish_reason("stop_sequence"), "stop");
    }

    #[test]
    fn passes_through_unrecognized_values() {
        assert_eq!(openai_finish_reason_to_anthropic_stop_reason("weird"), "weird");
        assert_eq!(anthropic_stop_reason_to_openai_finish_reason("weird"), "weird");
    }
}
