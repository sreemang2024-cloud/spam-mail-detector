"""
Generates data/sample_dataset.csv — a small, template-based spam/ham dataset
so the app has something to train on before you plug in a real Kaggle dataset.
Run: python data/generate_sample_dataset.py
"""
import csv
import os
import random

random.seed(42)

SPAM_TEMPLATES = [
    "CONGRATULATIONS! You have WON a ${amount} prize! Click here now to claim: {url}",
    "URGENT: Your account will be suspended. Verify your identity immediately at {url}",
    "Limited time offer! Get {amount}% off all products, buy now before it's gone!",
    "You are pre-approved for a $50000 loan. No credit check required, apply now!",
    "Work from home and earn $5000 per week! No experience needed, click {url}",
    "FREE gift card worth ${amount}! Just complete this short survey to claim yours.",
    "Your PayPal account has been LIMITED. Confirm your details now: {url}",
    "Hot singles in your area want to meet you tonight! Sign up free at {url}",
    "Act now! This offer expires in 24 hours. Call now to lock in your discount.",
    "You've been selected for a free iPhone! Claim your prize before it expires.",
    "Lose 20 pounds in 2 weeks with this ONE weird trick doctors don't want you to know!",
    "Your package could not be delivered. Update your payment info at {url} to reschedule.",
    "WINNER! Your email address has won the international lottery. Reply to claim ${amount}.",
    "Cheap meds online, no prescription needed! Order today and save {amount}%.",
    "Dear customer, unusual activity was detected. Verify your bank account now at {url}.",
    "Make money fast with this proven system! Guaranteed income of ${amount} monthly.",
    "Your subscription payment failed. Update billing info immediately to avoid suspension.",
    "Get rich quick with crypto trading bot! Turn $100 into ${amount} in one week.",
    "FINAL NOTICE: Your domain registration expires today. Renew now at {url}.",
    "Exclusive investment opportunity! Double your money guaranteed, risk-free returns.",
]

HAM_TEMPLATES = [
    "Hi {name}, just checking in on the {topic} report — do you have an update?",
    "Thanks for the meeting notes yesterday, really helpful ahead of the {topic} review.",
    "Can we move our {topic} call to 3pm tomorrow? Let me know if that works.",
    "Attached is the invoice for last month's {topic} services. Let me know if you need anything else.",
    "Reminder: the {topic} workshop starts at 10am in the main conference room.",
    "Hey, are we still on for lunch this week to talk about the {topic} project?",
    "Please review the attached document about {topic} and share your feedback by Friday.",
    "Great catching up at the {topic} conference — let's stay in touch.",
    "Here are the slides from today's {topic} presentation for your records.",
    "Following up on our conversation about {topic}. I've included the updated numbers.",
    "The {topic} deployment went smoothly, all tests passed in staging.",
    "Could you send me the latest version of the {topic} spreadsheet when you get a chance?",
    "Happy birthday! Hope you have a wonderful day with family and friends.",
    "Your order has shipped and should arrive within 3-5 business days.",
    "Team standup notes: {topic} is on track, no blockers reported this week.",
    "Thank you for your purchase. Your receipt for {topic} services is attached.",
    "Let's schedule a quick sync about the {topic} roadmap for next quarter.",
    "I've shared the {topic} folder with you on the drive, let me know if access works.",
    "Quick reminder that rent is due on the 1st of next month, thanks!",
    "The weather looks great this weekend, want to go hiking with the {topic} group?",
]

NAMES = ["Alex", "Priya", "Sam", "Jordan", "Maria", "Chen", "Fatima", "Liam"]
TOPICS = ["budget", "marketing", "onboarding", "design", "engineering", "sales", "product", "billing"]
URLS = ["bit.ly/claim-now", "secure-verify-account.com", "win-big-today.net", "quickcash-offer.com"]


def build_rows(n_each: int = 150):
    rows = []
    for _ in range(n_each):
        t = random.choice(SPAM_TEMPLATES)
        text = t.format(amount=random.choice([100, 250, 500, 1000, 5000]), url=random.choice(URLS))
        rows.append(("spam", text))
    for _ in range(n_each):
        t = random.choice(HAM_TEMPLATES)
        text = t.format(name=random.choice(NAMES), topic=random.choice(TOPICS))
        rows.append(("ham", text))
    random.shuffle(rows)
    return rows


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_dataset.csv')
    rows = build_rows()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
