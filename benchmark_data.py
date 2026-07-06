"""
ShopEase RAG Benchmark – Ground-Truth Dataset
20 query/answer pairs covering all 5 categories.
"""

# ---------------------------------------------------------------------------
# Pricing (USD per million tokens)
# ---------------------------------------------------------------------------
ANTHROPIC_PRICING = {
    "claude-sonnet-4-6": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
    }
}

# ---------------------------------------------------------------------------
# Ground-truth test cases
# Each entry:
#   query               – the customer question to pose
#   expected_doc_ids    – list of doc IDs that should be retrieved (order matters for MRR/NDCG)
#   expected_answer_topics – keywords the answer must touch (used by LLM judge as hints)
#   category            – one of: returns, shipping, payment, account, warranty
#   should_escalate     – True if the correct agent behaviour is to escalate
# ---------------------------------------------------------------------------
GROUND_TRUTH = [
    # ── Returns (4) ──────────────────────────────────────────────────────────
    {
        "query": "What is your return policy? How many days do I have to return something?",
        "expected_doc_ids": ["RET-001"],
        "expected_answer_topics": ["30 days", "unused", "original packaging", "5-7 business days", "refund"],
        "category": "returns",
        "should_escalate": False,
    },
    {
        "query": "Can I return a digital download I purchased last week?",
        "expected_doc_ids": ["RET-002"],
        "expected_answer_topics": ["non-returnable", "digital downloads", "final sale"],
        "category": "returns",
        "should_escalate": False,
    },
    {
        "query": "I received the wrong size. How do I exchange it for a larger one?",
        "expected_doc_ids": ["RET-003"],
        "expected_answer_topics": ["exchange", "return", "new order", "defective", "replacement"],
        "category": "returns",
        "should_escalate": False,
    },
    {
        "query": "I sent back my item two weeks ago but still haven't received my refund. This is unacceptable!",
        "expected_doc_ids": ["RET-004", "RET-001"],
        "expected_answer_topics": ["email notification", "5-7 business days", "original payment method", "bank"],
        "category": "returns",
        "should_escalate": True,
    },

    # ── Shipping (4) ─────────────────────────────────────────────────────────
    {
        "query": "How much does shipping cost and how long will it take?",
        "expected_doc_ids": ["SHIP-001"],
        "expected_answer_topics": ["standard", "free", "$35", "expedited", "overnight", "business days"],
        "category": "shipping",
        "should_escalate": False,
    },
    {
        "query": "Do you ship to Canada? How long does international delivery take?",
        "expected_doc_ids": ["SHIP-002"],
        "expected_answer_topics": ["50 countries", "7-21 business days", "customs", "import taxes"],
        "category": "shipping",
        "should_escalate": False,
    },
    {
        "query": "I got a shipping confirmation but my tracking number doesn't show any updates.",
        "expected_doc_ids": ["SHIP-003"],
        "expected_answer_topics": ["tracking number", "24 hours", "carrier", "UPS", "FedEx", "USPS"],
        "category": "shipping",
        "should_escalate": False,
    },
    {
        "query": "My package arrived completely crushed and the item inside is broken.",
        "expected_doc_ids": ["SHIP-004"],
        "expected_answer_topics": ["damaged", "photos", "packaging", "claim", "replacement", "refund"],
        "category": "shipping",
        "should_escalate": True,
    },

    # ── Payment (4) ──────────────────────────────────────────────────────────
    {
        "query": "What payment methods do you accept? Do you take Apple Pay?",
        "expected_doc_ids": ["PAY-001"],
        "expected_answer_topics": ["Visa", "Mastercard", "PayPal", "Apple Pay", "Google Pay", "SSL"],
        "category": "payment",
        "should_escalate": False,
    },
    {
        "query": "I have a promo code but it doesn't seem to work at checkout.",
        "expected_doc_ids": ["PAY-002"],
        "expected_answer_topics": ["promo code", "checkout", "one promo code", "Rewards points"],
        "category": "payment",
        "should_escalate": False,
    },
    {
        "query": "Can I split my $200 order into smaller payments over time?",
        "expected_doc_ids": ["PAY-003"],
        "expected_answer_topics": ["Klarna", "Afterpay", "4", "interest-free", "$30", "$1,500"],
        "category": "payment",
        "should_escalate": False,
    },
    {
        "query": "My card keeps getting declined even though I know there's money in my account.",
        "expected_doc_ids": ["PAY-004"],
        "expected_answer_topics": ["card number", "expiry", "CVV", "billing address", "bank"],
        "category": "payment",
        "should_escalate": False,
    },

    # ── Account (4) ──────────────────────────────────────────────────────────
    {
        "query": "I forgot my password and can't log in to my ShopEase account.",
        "expected_doc_ids": ["ACCT-001"],
        "expected_answer_topics": ["Forgot Password", "email", "reset link", "1 hour", "spam"],
        "category": "account",
        "should_escalate": False,
    },
    {
        "query": "I think someone has hacked my account. What should I do?",
        "expected_doc_ids": ["ACCT-002", "ACCT-001"],
        "expected_answer_topics": ["two-factor authentication", "2FA", "change your password", "security team"],
        "category": "account",
        "should_escalate": True,
    },
    {
        "query": "How does the ShopEase Rewards program work? How do I redeem my points?",
        "expected_doc_ids": ["ACCT-003"],
        "expected_answer_topics": ["1 point per dollar", "100 points", "$1", "Gold", "expedited shipping", "12 months"],
        "category": "account",
        "should_escalate": False,
    },
    {
        "query": "I want to permanently delete my ShopEase account. Will I lose my rewards points?",
        "expected_doc_ids": ["ACCT-004"],
        "expected_answer_topics": ["Close Account", "forfeit", "Rewards points", "7 years", "data retention"],
        "category": "account",
        "should_escalate": False,
    },

    # ── Warranty (4) ─────────────────────────────────────────────────────────
    {
        "query": "My new laptop stopped working after 3 months. Is it covered under warranty?",
        "expected_doc_ids": ["WARR-001"],
        "expected_answer_topics": ["manufacturer's warranty", "1-2 years", "defects", "workmanship", "warranty claim"],
        "category": "warranty",
        "should_escalate": False,
    },
    {
        "query": "What does the ShopEase extended warranty cover and how much does it cost?",
        "expected_doc_ids": ["WARR-002"],
        "expected_answer_topics": ["ShopEase Protect", "1-3 years", "accidental damage", "$7.99", "replace", "refund"],
        "category": "warranty",
        "should_escalate": False,
    },
    {
        "query": "I accidentally dropped my tablet and cracked the screen. Does the warranty cover accidental damage?",
        "expected_doc_ids": ["WARR-001", "WARR-002"],
        "expected_answer_topics": ["accidental damage", "manufacturer's warranty", "ShopEase Protect", "extended"],
        "category": "warranty",
        "should_escalate": False,
    },
    {
        "query": "How do I file a warranty claim for a defective item I bought six months ago?",
        "expected_doc_ids": ["WARR-001", "WARR-002"],
        "expected_answer_topics": ["warranty claim", "manufacturer", "shopease.com/protect", "repair", "replace"],
        "category": "warranty",
        "should_escalate": False,
    },
]
