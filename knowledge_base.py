"""
ShopEase Customer Support Knowledge Base
Loads FAQ documents into a ChromaDB in-memory vector store.
"""

import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# FAQ documents – 18 realistic entries for ShopEase
# ---------------------------------------------------------------------------
FAQ_DOCS = [
    # ── Returns ──────────────────────────────────────────────────────────────
    {
        "doc_id": "RET-001",
        "category": "returns",
        "text": (
            "ShopEase Return Policy: You may return most items within 30 days of delivery "
            "for a full refund. Items must be unused, in original packaging, and accompanied "
            "by the original receipt or order confirmation. To start a return, visit "
            "shopease.com/returns or contact support. Refunds are processed within 5-7 "
            "business days after we receive the item."
        ),
    },
    {
        "doc_id": "RET-002",
        "category": "returns",
        "text": (
            "Non-Returnable Items at ShopEase: The following items cannot be returned – "
            "perishable goods (food, flowers), digital downloads, personalized/custom-made "
            "products, intimate apparel, and hazardous materials. Gift cards and "
            "clearance items marked 'Final Sale' are also non-returnable."
        ),
    },
    {
        "doc_id": "RET-003",
        "category": "returns",
        "text": (
            "ShopEase Exchanges: To exchange an item for a different size or colour, "
            "initiate a return for the original item and place a new order for the "
            "replacement. We do not currently support direct exchanges. If you received "
            "a defective item, contact support within 7 days and we will send a free "
            "replacement with priority shipping."
        ),
    },
    {
        "doc_id": "RET-004",
        "category": "returns",
        "text": (
            "ShopEase Refund Status: Once your returned item is received and inspected, "
            "you will receive an email notification. Approved refunds are credited to the "
            "original payment method within 5-7 business days. Credit card refunds may "
            "take an additional 3-5 business days to appear on your statement depending "
            "on your bank."
        ),
    },
    # ── Shipping ─────────────────────────────────────────────────────────────
    {
        "doc_id": "SHIP-001",
        "category": "shipping",
        "text": (
            "ShopEase Shipping Options: Standard shipping (5-7 business days) is free on "
            "orders over $35. Expedited shipping (2-3 business days) costs $9.99. Overnight "
            "shipping is available for $24.99. Orders placed before 2 PM EST on business days "
            "are typically processed the same day."
        ),
    },
    {
        "doc_id": "SHIP-002",
        "category": "shipping",
        "text": (
            "ShopEase International Shipping: We ship to over 50 countries. International "
            "shipping rates and delivery times vary by destination. Estimated delivery is "
            "7-21 business days for international orders. Customers are responsible for "
            "any customs duties or import taxes imposed by their country."
        ),
    },
    {
        "doc_id": "SHIP-003",
        "category": "shipping",
        "text": (
            "Tracking Your ShopEase Order: Once your order ships, you will receive a "
            "tracking number via email. You can track your package at shopease.com/track "
            "or through the carrier's website (UPS, FedEx, or USPS). Tracking information "
            "may take up to 24 hours to update after shipment."
        ),
    },
    {
        "doc_id": "SHIP-004",
        "category": "shipping",
        "text": (
            "ShopEase Lost or Damaged Packages: If your package appears lost (no movement "
            "for 5+ business days) or arrives damaged, contact ShopEase support immediately. "
            "For damaged deliveries, take photos before opening fully and keep all packaging. "
            "We will file a claim with the carrier and send a replacement or issue a refund "
            "within 3-5 business days."
        ),
    },
    # ── Payment ──────────────────────────────────────────────────────────────
    {
        "doc_id": "PAY-001",
        "category": "payment",
        "text": (
            "ShopEase Payment Methods: We accept Visa, Mastercard, American Express, "
            "Discover, PayPal, Apple Pay, Google Pay, and ShopEase Gift Cards. "
            "All transactions are secured with 256-bit SSL encryption. We do not store "
            "full credit card numbers on our servers."
        ),
    },
    {
        "doc_id": "PAY-002",
        "category": "payment",
        "text": (
            "ShopEase Promo Codes & Discounts: To apply a promo code, enter it in the "
            "'Promo Code' field at checkout and click Apply. Only one promo code may be "
            "used per order. Promo codes cannot be applied after an order is placed. "
            "ShopEase Rewards points can be redeemed alongside promo codes."
        ),
    },
    {
        "doc_id": "PAY-003",
        "category": "payment",
        "text": (
            "ShopEase Buy Now Pay Later: ShopEase offers Buy Now Pay Later through Klarna "
            "and Afterpay. Split your purchase into 4 interest-free payments. Available "
            "for orders between $30 and $1,500. Subject to approval and credit check "
            "by the BNPL provider. ShopEase is not responsible for BNPL provider terms."
        ),
    },
    {
        "doc_id": "PAY-004",
        "category": "payment",
        "text": (
            "ShopEase Payment Declined: If your payment is declined, verify that the "
            "card number, expiry date, and CVV are correct and that your billing address "
            "matches your card on file. Ensure you have sufficient funds. Try a different "
            "payment method or contact your bank. Our system does not store declined "
            "card details."
        ),
    },
    # ── Account Issues ───────────────────────────────────────────────────────
    {
        "doc_id": "ACCT-001",
        "category": "account",
        "text": (
            "ShopEase Account Password Reset: To reset your password, click 'Forgot Password' "
            "on the login page and enter your email address. You will receive a reset link "
            "within 5 minutes. The link expires in 1 hour. If you do not receive the email, "
            "check your spam folder or contact support."
        ),
    },
    {
        "doc_id": "ACCT-002",
        "category": "account",
        "text": (
            "ShopEase Account Security: We recommend enabling two-factor authentication (2FA) "
            "in your account settings for additional security. ShopEase will never ask for "
            "your password via email or phone. If you suspect unauthorized access to your "
            "account, change your password immediately and contact our security team."
        ),
    },
    {
        "doc_id": "ACCT-003",
        "category": "account",
        "text": (
            "ShopEase Rewards Program: ShopEase Rewards members earn 1 point per dollar "
            "spent. Points can be redeemed for discounts (100 points = $1 off). "
            "Gold members (500+ points/year) get free expedited shipping and early "
            "access to sales. Points expire after 12 months of account inactivity."
        ),
    },
    {
        "doc_id": "ACCT-004",
        "category": "account",
        "text": (
            "Closing Your ShopEase Account: To permanently close your account, go to "
            "Account Settings > Privacy > Close Account. Note that closing your account "
            "will forfeit any unredeemed Rewards points and Gift Card balances. "
            "Order history will be retained for 7 years for legal/tax purposes per "
            "our data retention policy."
        ),
    },
    # ── Warranty ─────────────────────────────────────────────────────────────
    {
        "doc_id": "WARR-001",
        "category": "warranty",
        "text": (
            "ShopEase Product Warranty: Most electronics sold on ShopEase come with a "
            "manufacturer's warranty of 1-2 years covering defects in materials and "
            "workmanship. Warranty does not cover accidental damage, misuse, or normal "
            "wear and tear. To make a warranty claim, contact the manufacturer directly "
            "or use ShopEase's warranty assistance service."
        ),
    },
    {
        "doc_id": "WARR-002",
        "category": "warranty",
        "text": (
            "ShopEase Extended Warranty (ShopEase Protect): ShopEase Protect plans extend "
            "coverage by 1-3 years beyond the manufacturer's warranty and include accidental "
            "damage protection. Plans start at $7.99/month. Claims can be filed at "
            "shopease.com/protect. If a repair is not possible, we will replace or refund "
            "the item."
        ),
    },
]


# ---------------------------------------------------------------------------
# Knowledge base initialisation
# ---------------------------------------------------------------------------

def build_knowledge_base():
    """
    Create an in-memory ChromaDB collection and populate it with FAQ docs.
    Uses SentenceTransformer embeddings via ChromaDB's built-in EF.
    Returns the collection object.
    """
    client = chromadb.Client()  # in-memory, no persistence

    # Use the sentence-transformers embedding function bundled with chromadb
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete collection if it already exists (for idempotent re-runs)
    try:
        client.delete_collection("shopease_faq")
    except Exception:
        pass

    collection = client.create_collection(
        name="shopease_faq",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch-upsert all documents
    collection.upsert(
        ids=[doc["doc_id"] for doc in FAQ_DOCS],
        documents=[doc["text"] for doc in FAQ_DOCS],
        metadatas=[
            {"category": doc["category"], "doc_id": doc["doc_id"]}
            for doc in FAQ_DOCS
        ],
    )

    return collection


def search_knowledge_base_raw(collection, query: str, n_results: int = 3):
    """
    Perform a semantic search against the ChromaDB collection.

    Returns a list of dicts with keys: doc_id, category, text, distance.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, len(FAQ_DOCS)),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "doc_id": meta.get("doc_id", "unknown"),
                    "category": meta.get("category", "general"),
                    "text": doc,
                    "distance": round(dist, 4),
                }
            )
    return output


def get_kb_stats(collection) -> dict:
    """Return basic statistics about the knowledge base."""
    count = collection.count()
    categories = {}
    for doc in FAQ_DOCS:
        cat = doc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    return {
        "total_documents": count,
        "categories": categories,
    }
