"""
Reflection Loop - Multi-pass self-critique for higher quality answers.

After the agent produces an initial answer, runs it through structured
reflection passes to catch errors, gaps, and non-standard approaches
before the final response is sent to the user.

Also includes self-consistency sampling for factual/reasoning questions.
"""

import logging

log = logging.getLogger(__name__)


SELF_CONSISTENCY_PROMPT = """You were asked the same question 3 times and gave these answers:

Answer 1: {a1}

Answer 2: {a2}

Answer 3: {a3}

Compare all three answers. Which one is most accurate and complete?
If they agree, use the consensus. If they disagree, pick the best one and explain why.
Write the final definitive answer."""


def self_consistency_sample(agent: object, question: str, context: str = "", n: int = 3) -> str:
    """
    Self-consistency sampling: ask the same question N times with higher
    temperature to get diverse reasoning paths, then pick the best answer.

    Research shows +17.9% accuracy on math, +11% on reasoning benchmarks.
    """
    from .core import Agent, AgentConfig

    samples = []
    for i in range(n):
        try:
            # Create a fresh agent each time to avoid history contamination
            sample_agent = Agent(AgentConfig(
                model=agent.config.model,
                temperature=0.7,  # Higher temp for diverse paths
                system_prompt="You are a helpful assistant. Be concise and accurate. Think step by step.",
            ))
            # Copy tools from main agent
            for tool in agent.tools:
                sample_agent.register_tool(tool)

            result = sample_agent.run(question, context)
            if result and not result.startswith("Agent error:"):
                samples.append(result)
                log.debug(f"[SelfConsistency] Sample {i+1}/{n}: {result[:80]}...")
        except Exception as e:
            log.warning(f"[SelfConsistency] Sample {i+1} failed: {e}")

    if not samples:
        return ""

    if len(samples) == 1:
        return samples[0]

    # Have the main agent pick the best answer
    try:
        consensus = agent.chat(SELF_CONSISTENCY_PROMPT.format(
            a1=samples[0][:1000],
            a2=samples[1][:1000] if len(samples) > 1 else "(same as Answer 1)",
            a3=samples[2][:1000] if len(samples) > 2 else "(same as Answer 1)",
        ))
        # Restore history (don't pollute with consistency prompt)
        if hasattr(agent, "messages") and agent.messages:
            saved = agent.messages[:-2]  # remove the consistency Q&A
            agent.messages = saved
        return consensus
    except Exception as e:
        log.warning(f"[SelfConsistency] Consensus failed: {e}")
        return samples[0]  # fallback to first sample

# Reflection prompts run sequentially after the initial answer.
# Each builds on the conversation history so the model sees its own prior work.
REFLECTION_PASSES = [
    (
        "intent",
        """Before evaluating your answer, step back and consider:
- Why is this person asking this question? What problem are they actually trying to solve?
- Are they a beginner looking for guidance, or an expert looking for a specific detail?
- Is there an underlying need or concern behind the question that they didn't explicitly state?
- Could the question be an XY problem — are they asking about their attempted solution when the real issue is something else?

Identify the true intent so your answer addresses what they actually need, not just what they literally asked.""",
    ),
    (
        "completeness",
        """Review your last answer critically. Ask yourself:
- Does it fully answer what was asked, or did you miss part of the question?
- Are there important caveats, edge cases, or gotchas you didn't mention?
- Is anything vague that should be specific?

DO NOT repeat your answer yet. Just identify what's missing or weak, if anything. Be honest.""",
    ),
    (
        "best_practices",
        """Now evaluate whether your approach is the industry-standard, best-practice way:
- Is this how senior engineers / experts actually do it?
- Are there better libraries, patterns, or techniques you should have recommended?
- Would a code reviewer push back on anything?

Again, just identify issues — don't rewrite the answer yet.""",
    ),
    (
        "adversarial",
        """Play devil's advocate. Steelman the strongest possible objection to your answer:
- What would a skeptic, expert, or critic say is wrong or oversimplified?
- Are there real-world scenarios where your advice would fail or backfire?
- Is there an entirely different approach that contradicts yours but is also valid?

Be genuinely critical — not just nitpicky. If your answer holds up, say so and why.""",
    ),
    (
        "final",
        """IMPORTANT: Everything above was YOUR OWN internal self-critique. The user did NOT say any of it.
The user has NOT pushed back, disagreed, or corrected you. Do NOT reference your self-critique or say things like "you're right to push back" — the user never pushed back.

Now write the definitive final answer to the user's ORIGINAL question.
- Address their true intent (not just the literal question)
- Silently incorporate improvements from your self-review — don't mention that you reviewed yourself
- Write as if this is your first and only response to the user
- Be thorough where depth matters, concise where it doesn't.""",
    ),
]


# Light mode only runs completeness + final (skip intent, best_practices, adversarial)
LIGHT_PASSES = ["completeness", "final"]


def reflect(
    agent: object, original_question: str, mode: str = "full"
) -> tuple[str, list[tuple[str, str]]]:
    """
    Run reflection passes on the agent's last answer.

    Args:
        agent: The Agent instance (must have already produced an initial answer via run())
        original_question: The user's original question (used for logging only)
        mode: "full" for all 5 passes, "light" for completeness + final only,
              "factual" for fact-check + final

    Returns:
        Tuple of (final_answer, thoughts) where thoughts is a list of
        (pass_name, thinking_text) for each pass that produced thinking.
    """
    if mode == "light":
        passes = [(n, p) for n, p in REFLECTION_PASSES if n in LIGHT_PASSES]
    elif mode == "factual":
        passes = FACTUAL_PASSES
    else:
        passes = REFLECTION_PASSES

    # Save the conversation history BEFORE reflection so we can restore it after.
    # Without this, reflection prompts pollute the agent's message history and
    # subsequent user messages see all the self-critique as part of the conversation.
    saved_messages = agent.messages.copy() if hasattr(agent, "messages") else None

    final_answer = ""
    thoughts: list[tuple[str, str]] = []

    for pass_name, prompt in passes:
        log.debug(f"[Reflection] Running pass: {pass_name}")
        try:
            result = agent.chat(prompt)
            thinking = getattr(agent, "last_thinking", "")
            if thinking:
                thoughts.append((pass_name, thinking))
            if pass_name == "final":
                final_answer = result
        except Exception as e:
            log.warning(f"[Reflection] Pass '{pass_name}' failed: {e}")
            break

    # Restore the original conversation history, replacing the agent's last
    # response with the refined final answer so subsequent messages see the
    # improved version without all the reflection noise.
    if saved_messages is not None and final_answer:
        # Find the last assistant message and replace its content with the refined answer
        for i in range(len(saved_messages) - 1, -1, -1):
            if saved_messages[i].get("role") == "assistant":
                saved_messages[i] = {"role": "assistant", "content": final_answer}
                break
        agent.messages = saved_messages

    return final_answer, thoughts


FACT_CHECK_PROMPT = """Review your last answer for factual accuracy. Ask yourself:
- Did you state any facts (geography, history, statistics, dates, people, science) without verifying them?
- Did you make any claims that could be wrong based on outdated or incorrect training data?
- Did you state anything confidently that you're not actually 100% certain about?

If you find ANY unverified factual claims, call web_search to verify them NOW.
If you already used web_search and have sources, confirm they support your claims.
If something is wrong, correct it. If you can't verify, flag it with "I'm not certain about this."

List what you verified and what needs correction. Be ruthlessly honest."""

# Factual mode: fact-check + final (verify claims with web search)
FACTUAL_PASSES = [
    ("fact_check", FACT_CHECK_PROMPT),
    REFLECTION_PASSES[-1],  # final pass
]


def auto_search_for_factual(agent: object, question: str) -> str:
    """
    For factual questions, search Wikipedia + web BEFORE the model answers.

    Returns search results string to inject into the user message (not system
    context — models ignore system context too easily).
    Has a 10-second timeout to prevent blocking.
    """
    import concurrent.futures

    from .utility_tools import wikipedia_summary
    from .web_search import web_search

    parts = []

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run Wikipedia and web search in parallel
            wiki_future = executor.submit(wikipedia_summary, question)
            web_future = executor.submit(web_search, question, 3)

            # Wikipedia first (more reliable)
            try:
                wiki = wiki_future.result(timeout=10)
                if wiki and "No Wikipedia article" not in wiki and "error" not in wiki.lower():
                    parts.append(f"WIKIPEDIA:\n{wiki}")
            except Exception:
                pass

            # Web search as supplement
            try:
                web = web_future.result(timeout=10)
                if web and "No results found" not in web and "Search error" not in web:
                    parts.append(f"WEB SEARCH:\n{web}")
            except Exception:
                pass

    except Exception as e:
        log.warning(f"[FactCheck] Auto-search failed: {e}")

    if parts:
        log.info(f"[FactCheck] Auto-searched: {question[:60]}...")
        return "\n\n".join(parts)
    return ""


SUFFICIENCY_PROMPT = """Look at your last response and the user's original question.
Does your response directly answer the question? If so, rewrite your response with ONLY the direct answer.
Remove all tangents, personal info dumps, profile summaries, and anything not directly asked for.
If the user asked "what time is it?" — just give the time. Nothing else.
Keep your tone natural but be concise. One or two sentences max for simple factual questions."""


def trim_response(agent: object) -> str:
    """
    Run a single sufficiency pass to trim bloated responses for simple questions.
    Returns the trimmed answer. Does NOT pollute conversation history.
    """
    saved_messages = agent.messages.copy() if hasattr(agent, "messages") else None
    try:
        result = agent.chat(SUFFICIENCY_PROMPT)
        # Restore history with trimmed answer replacing the original
        if saved_messages is not None and result:
            for i in range(len(saved_messages) - 1, -1, -1):
                if saved_messages[i].get("role") == "assistant":
                    saved_messages[i] = {"role": "assistant", "content": result}
                    break
            agent.messages = saved_messages
        return result
    except Exception as e:
        log.warning(f"[Sufficiency] Trim pass failed: {e}")
        if saved_messages is not None:
            agent.messages = saved_messages
        return ""


import re


def classify_question(message: str, response: str) -> str:
    """
    Classify a question to determine how much reflection it needs.

    Returns one of:
        "skip"    - No reflection. Greetings, casual chat, bot commands.
        "factual" - Factual question. Auto web search + fact-check pass. No deep reflection.
        "light"   - Light reflection (completeness + final only). Simple technical questions.
        "full"    - Full 5-pass reflection. Ideological, architectural, opinion, strategy questions.

    The goal: don't waste 5 GPU passes on "what time is it?" but do invest
    them on "how should I architect my microservices?" And for factual questions,
    always verify with a web search before answering.
    """
    lowered = message.strip().lower()
    stripped = message.strip()

    # --- SKIP: Casual greetings and acknowledgments ---
    casual_patterns = (
        "hey", "hi", "hello", "thanks", "thank you", "lol", "haha",
        "ok", "okay", "cool", "nice", "awesome", "got it", "sounds good",
        "yes", "no", "yep", "nope", "sure", "bye", "gn", "gm",
        "ty", "np", "gg", "brb", "wb",
    )
    # Only match if the entire message is casual (not just starts with it)
    words = lowered.split()
    if len(words) <= 4 and any(lowered.startswith(s) for s in casual_patterns):
        # But don't skip if it contains a question mark — could be a real question
        if "?" not in stripped:
            return "skip"

    # --- SKIP: Very short responses (bot had nothing substantive to say) ---
    if len(response.strip()) < 100:
        return "skip"

    # --- SKIP: Bot commands (not real questions) ---
    command_patterns = [
        r"^(show|list|tell me) (my|the) (enhancements|tools|commands)",
        r"^(start|stop|restart|status)\b",
        r"^remind me\b",
    ]
    if any(re.search(p, lowered) for p in command_patterns):
        return "skip"

    # --- SKIP: Time/date (bot can answer from system clock, no search needed) ---
    time_patterns = [
        r"^what (time|day|date)",
        r"^what is (today|the date|the time)",
    ]
    if any(re.search(p, lowered) for p in time_patterns):
        return "skip"

    # --- FACTUAL: Verifiable questions that need web search ---
    factual_patterns = [
        r"^(where) (is|are|was|were|did)\b",
        r"^who (is|are|was|were)\b",
        r"^(when) (is|are|was|were|did)\b",
        r"^how (old|tall|long|far|much does .* (weigh|cost))",
        r"^(define|meaning of|what does .* mean)",
        r"^(convert|translate)\b",
        r"^what is \d",  # math questions like "what is 2+2"
        r"\b(where is|located|capital of|population of|how far)\b",
        r"\b(who (invented|created|founded|discovered|wrote|directed))\b",
        r"\b(when did|what year|how old is)\b",
        r"\b(how (much|many|big|small|fast|slow) (is|are|was|were))\b",
        r"^what is (the |a )?\w",  # "what is X" pattern
        r"^(tell me about|what do you know about)\b",
    ]
    if any(re.search(p, lowered) for p in factual_patterns):
        return "factual"

    # --- FULL: Deep thinking questions ---
    deep_patterns = [
        r"\b(should i|would you recommend|what do you think|opinion|advice)\b",
        r"\b(how (should|would|could) (i|we|you))\b",
        r"\b(best (way|approach|practice|strategy))\b",
        r"\b(pros? and cons?|trade.?offs?|compared? to|versus|vs\.?)\b",
        r"\b(architect|design|structure|organize|plan)\b",
        r"\b(future|impact|affect|implications?|consequences?)\b",
        r"\b(why (is|are|do|does|did|should|would|can))\b",
        r"\b(ethical|moral|philosophical|ideological)\b",
        r"\b(strategy|roadmap|long.?term|big picture)\b",
        r"\b(explain|walk me through|deep dive|in depth)\b",
        r"\b(debug|troubleshoot|figure out|diagnose)\b",
        r"\b(review|critique|evaluate|assess)\b",
    ]
    if any(re.search(p, lowered) for p in deep_patterns):
        return "full"

    # --- LIGHT: Everything else (simple technical, short how-to, etc.) ---
    # If the response is long, upgrade to full (model had a lot to say)
    if len(response.strip()) > 500:
        return "full"

    return "light"


def is_factual_question(message: str) -> bool:
    """
    Quick check if a question is factual (needs web search) based on the
    question alone, without needing the response. Used for pre-classification
    before the agent answers.
    """
    lowered = message.strip().lower()

    # Only search for questions that NEED current/specific data.
    # Don't search for basic knowledge ("what is spaghetti") — the LLM knows that.
    factual_patterns = [
        r"\b(capital of|population of|how far from)\b",
        r"\b(who (invented|created|founded|discovered|wrote|directed))\b",
        r"\b(when did|what year)\b",
        r"\b(current (price|stock|weather|score|status))\b",
        r"\b(latest|newest|recent|today's|this week's)\b",
        r"\b(how much does .* (cost|weigh))\b",
        r"\b(release date|launched|announced)\b",
    ]
    return any(re.search(p, lowered) for p in factual_patterns)


