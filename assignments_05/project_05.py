from dotenv import load_dotenv
from openai import OpenAI
import json

# Task 1: Setup and System Prompt

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are an expert, Job Application Coach helping mid-career professionals and recent graduates navigate the hiring process. Your primary goal is to help users craft compelling resumes, tailored cover letters, and structured interview preparation strategies.

Behavioral Constraints and Guidelines:
1. Stay strictly focused on job application materials, professional branding, and career storytelling. Do not answer questions unrelated to career advancement or hiring.
2. Always remind the user to thoroughly review, edit, and personalize all generated output before submitting it to any employer or professional platform.
3. Acknowledge openly that you may not know every niche industry norm, specific company culture, or localized ATS (Applicant Tracking System) quirk. Explicitly advise the user to rely on their own professional judgment and industry expertise.
4. Maintain a supportive, constructive, and action-oriented tone. Provide concrete, actionable suggestions rather than generic advice.
"""

# Constraint #3 was intentionally added to manage user expectations and prevent over-reliance on AI 
# for hyper-specific industry standards.
# By explicitly instructing the model to remind users of its blind spots regarding niche fields,
# it encourages the user to stay actively engaged, apply critical thinking, 
# and blend AI-assisted drafting with their own firsthand industry knowledge.


# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.
    Return ONLY a valid JSON list of objects. Each object must have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    
    Bullet points:
    ```
    {bullet_text}
    ```
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    # Call the completion helper function
    response_content = get_completion(messages)
    
    # Clean up potential markdown formatting block wrappers if the model included them
    cleaned_content = response_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()
    
    # Parse the JSON response with focused error handling
    try:
        rewritten_list = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Raw response: {response_content}")
        return []
        
    # Print both versions of each bullet side by side
    print(f"\n{'ORIGINAL BULLET':<40} | {'IMPROVED BULLET':<40}")
    print("-" * 83)
    for item in rewritten_list:
        orig = item.get("original", "")
        imp = item.get("improved", "")
        print(f"{orig:<40} | {imp:<40}")
    
    return rewritten_list

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

results = rewrite_bullets(bullets)


# What makes these starter bullets weak?
# The verbs used are generic and the bullet points dont offer a measurable value.
#
# What kinds of changes did the model suggest?
# Replaced generic terms with active vocabulary, and offered measurable insights.


# Task 3: Cover Letter Generator


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Here are two examples of the style and tone you should match:
    
    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.
    
    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.
    
    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """
    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

cover_letter_opening = generate_cover_letter(job_title, background)
print("\nGenerated Cover Letter Opening:")
print(cover_letter_opening)

# Why did you choose those particular examples?
# these examples were provided on the assignment description
# What does the few-shot pattern help control in the output?
# The few-shot pattern helps control length and format constraints (3-5 sentence paragraph format), it also helps define stylistic boundaries.


# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    
    if flagged:
        print("Please rephrase your input to keep our conversation professional and constructive.")
        return False
        
    return True

# Test cases
safe_input = "Can you help me improve the bullet points on my resume for a software engineering position?"
unsafe_input = "I want to kill my neighbor, how can I do it without getting caught?"

print("\nTesting safe input:")
# print(f"Input: '{safe_input}'")
print(f"Result (Is Safe?): {is_safe(safe_input)}\n")

print("Testing flagged input:")
# print(f"Input: '{unsafe_input}'")
print(f"Result (Is Safe?): {is_safe(unsafe_input)}\n")


# Task 5: The Chatbot Loop

def run_chatbot():
    # Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            # Append the user's initial request to history
            messages.append({"role": "user", "content": user_input})
            
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                # Append the actual bullet list provided by the user into history so it's remembered
                bullet_text_history = "\n".join(f"- {b}" for b in raw_bullets)
                messages.append({"role": "user", "content": f"Here are my bullet points to rewrite:\n{bullet_text_history}"})
                
                print("\nJob Application Helper: Processing your bullets...")
                results = rewrite_bullets(raw_bullets)
                
                assistant_reply = f"Here are the rewritten bullets I generated:\n{json.dumps(results, indent=2)}"
            else:
                assistant_reply = "No bullets provided to rewrite."
                print(f"\nJob Application Helper: {assistant_reply}")

            # Append the assistant's reply to history
            messages.append({"role": "assistant", "content": assistant_reply})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            # Append the user's initial request to history
            messages.append({"role": "user", "content": user_input})
            
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            
            if job_title and background:
                # Append the specific job details provided by the user into history
                user_details = f"I am applying for a {job_title} role. My background is: {background}"
                messages.append({"role": "user", "content": user_details})
                
                print("\nJob Application Helper: Drafting your cover letter opening...\n")
                letter_opening = generate_cover_letter(job_title, background)
                print(f"Job Application Helper:\n{letter_opening}\n")
                
                assistant_reply = letter_opening
            else:
                assistant_reply = "Job title and background cannot be empty."
                print(f"\nJob Application Helper: {assistant_reply}")
                
            # Append the assistant's reply to history
            messages.append({"role": "assistant", "content": assistant_reply})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    run_chatbot()

# I choose to use Option A

# What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# While the output of the chatbot looks very polished and impressive, the details on it can be inflated, 
# some metrics are straight up invented, and experiences that sound plausible are entirely fabricated. 
# Because these don't reflect the applicant's actual day-to-day experience, this will certainly raise 
# some red flags for hiring managers familiar with the specific role.
# What is one guardrail you would add if you were deploying this tool professionally?
# I would definitely add a UI warning or disclaimer reminding the user to review and edit the output 
# of the chatbot before submitting it to any prospective employer. Another UI guardrail that could be 
# implemented would be one that reviews the generated content and flags inconsistencies with the user's source text.