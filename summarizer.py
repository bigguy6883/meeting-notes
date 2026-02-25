from groq import Groq

_client = Groq()

PROMPT_SIMPLE = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullet points of main topics discussed)
2. ACTION ITEMS (person: task, or "None identified" if none)
3. KEY DECISIONS (or "None identified" if none)

Transcript:
{transcript}"""

PROMPT_DIARIZED = """You are a meeting assistant. Analyze this transcript and provide:

1. SUMMARY (3-5 bullets — include specific numbers, dates, names, and commitments)

2. SPEAKERS — for each speaker:
   - Apparent role/name if identifiable
   - Interaction type: decision-maker | facilitator | questioner | contributor | dissenter
   - 1-line characterization of their style

3. KEY HIGHLIGHTS — concrete moments worth noting (specific quotes, commitments,
   surprises, or disagreements — not just topic labels)

4. ACTION ITEMS — person: task, deadline if mentioned (or "None identified")

5. KEY DECISIONS — what was decided and the brief rationale behind it

6. OPEN QUESTIONS — unresolved issues or follow-ups with no clear owner

Transcript:
{transcript}"""


def summarize(transcript, model="llama-3.3-70b-versatile", diarized=False):
    template = PROMPT_DIARIZED if diarized else PROMPT_SIMPLE
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": template.format(transcript=transcript)}]
    )
    return response.choices[0].message.content.strip()
