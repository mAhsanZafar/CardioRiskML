# ELA Professional – Shopify Theme

A complete custom Shopify theme for the **ELA Professional** cosmetics brand.  
Color scheme: 70% white / 30% black. Mobile-first. Clean, clinical, luxury aesthetic.

---

## File Structure

```
shopify-theme/
├── assets/
│   ├── ela-custom.css          # Full CSS: colors, typography, all components
│   ├── ela-cart-progress.js    # Cart drawer + free-shipping progress bar
│   └── ela-quiz.js             # Hair type quiz popup (3 questions)
├── config/
│   └── settings_schema.json    # Theme Editor settings (logo, colors, fonts)
├── layout/
│   └── theme.liquid            # Base HTML layout (header, footer, cart drawer)
├── sections/
│   ├── hero-banner.liquid      # Homepage hero image + headline
│   ├── category-cards.liquid   # SKINCARE / HAIRCARE two-column cards
│   ├── trust-bar.liquid        # Trust icons row (Sulfate-Free, Cruelty-Free…)
│   ├── featured-products.liquid # 4-product grid
│   └── bundles.liquid          # Bundles page section
├── snippets/
│   ├── product-ingredients.liquid  # Ingredient list from metafield
│   ├── product-how-to-use.liquid   # How-to-use steps from metafield
│   └── back-in-stock.liquid        # "Notify Me" form for out-of-stock products
└── templates/
    ├── index.liquid            # Homepage (assembles sections)
    ├── product.ela.liquid      # Product page (gallery, ATC, accordion, reviews)
    └── page.bundles.liquid     # Bundles page
```

---

## Deployment to Shopify

### Option A – Shopify CLI (recommended)

```bash
# 1. Install Shopify CLI
npm install -g @shopify/cli @shopify/theme

# 2. Authenticate
shopify auth login --store your-store.myshopify.com

# 3. Start with Dawn as the base theme, then overlay these files
shopify theme init ela-professional --clone-url https://github.com/Shopify/dawn
cd ela-professional

# 4. Copy all files from this directory into the Dawn theme
cp -r /path/to/shopify-theme/assets/*    assets/
cp -r /path/to/shopify-theme/sections/*  sections/
cp -r /path/to/shopify-theme/snippets/*  snippets/
cp -r /path/to/shopify-theme/templates/* templates/
cp -r /path/to/shopify-theme/layout/*    layout/
cp    /path/to/shopify-theme/config/settings_schema.json config/

# 5. Push to your Shopify store
shopify theme push

# 6. Preview in browser
shopify theme dev
```

### Option B – Shopify Admin (manual upload)

1. In Shopify Admin → **Online Store → Themes → Add theme → Upload ZIP**
2. Start from Dawn theme as base
3. Open Theme Editor → **Edit code**
4. Paste each file's contents into the corresponding theme file

---

## Product Setup

### Step 1 – Create Products

Create these 4 products in **Shopify Admin → Products**:

| Product | Handle | Collection |
|---------|--------|------------|
| Multi-Peptide Serum for Skin Density | `multi-peptide-serum-skin-density` | Skincare |
| Keratin Sulfate-Free Shampoo | `keratin-sulfate-free-shampoo` | Haircare |
| Keratin Hair Mask | `keratin-hair-mask` | Haircare |
| Multi-Peptide Serum for Hair Density | `multi-peptide-serum-hair-density` | Haircare |

Set **Product type** to `Skincare` or `Haircare` (displayed as collection label on the card).

### Step 2 – Assign Product Template

For each product: **Product page → Theme template → Select `product.ela`**

### Step 3 – Create Metafields

In **Shopify Admin → Settings → Custom Data → Products**, create:

| Namespace | Key | Type | Description |
|-----------|-----|------|-------------|
| `custom` | `short_description` | Single line text or Rich text | 1–2 sentence benefit summary |
| `custom` | `ingredients` | JSON | Array of `{ "name": "...", "benefit": "..." }` objects |
| `custom` | `how_to_use` | JSON | Array of strings (one per step) |

#### Example `ingredients` value:
```json
[
  { "name": "Multi-Peptide Complex", "benefit": "Signals skin to produce more collagen and improve density" },
  { "name": "Hyaluronic Acid", "benefit": "Attracts and retains moisture for plump, hydrated skin" },
  { "name": "Niacinamide (Vitamin B3)", "benefit": "Brightens tone and minimises pores" }
]
```

#### Example `how_to_use` value:
```json
[
  "Apply 3–4 drops to clean, dry skin morning and evening.",
  "Gently press into skin using fingertips — avoid rubbing.",
  "Follow with your moisturiser.",
  "Use consistently for best results. Visible improvement in 4–6 weeks."
]
```

---

## Collections Setup

Create two collections:

| Collection | Handle | Products |
|------------|--------|----------|
| Skincare | `skincare` | Multi-Peptide Serum for Skin Density |
| Haircare | `haircare` | Shampoo, Hair Mask, Hair Serum |

---

## Bundles Setup

1. Install the free **[Shopify Bundles](https://apps.shopify.com/shopify-bundles)** app
2. Create:
   - **Hair Repair Bundle** → Shampoo + Hair Mask + Hair Serum (15–17% discount)
   - **Skin & Hair Glow Bundle** → Skin Serum + Hair Serum (10–14% discount)
3. In Shopify Admin → **Pages → Add page**:
   - Title: `Bundles`
   - Template: `page.bundles`
4. Edit the `bundles` section via Theme Editor, link each bundle card to the bundle product

---

## Apps to Install

| App | Purpose | Plan |
|-----|---------|------|
| [Shopify Bundles](https://apps.shopify.com/shopify-bundles) | Bundle discounts | Free |
| [Judge.me Product Reviews](https://apps.shopify.com/judgeme) | Photo reviews | Free |
| [Klaviyo](https://apps.shopify.com/klaviyo-email-marketing) | Email + abandoned cart | Free ≤250 contacts |
| [Back In Stock – Restock Alerts](https://apps.shopify.com/back-in-stock) | Out-of-stock notifications | Free tier |

### Judge.me Setup
After installing, reviews appear automatically in the `#judgeme_product_reviews` div on the product page. No code changes needed.

### Klaviyo Abandoned Cart
1. Connect Klaviyo to Shopify
2. In Klaviyo → **Flows → Create Flow → Abandoned Cart**
3. Set delay: **1 hour**
4. Customize email: white background `#FFFFFF`, black text `#000000`, Montserrat font
5. Optional: include a 10% discount code for first recovery email

### Back In Stock
After installing, update the endpoint URL in `snippets/back-in-stock.liquid`:
```javascript
var endpoint = '/apps/back-in-stock/notify'; // ← replace with actual app endpoint
```
The app's documentation will provide the correct URL.

---

## Homepage Configuration (Theme Editor)

After deploying, in **Customize Theme**:

1. **Trust Bar** → 4 blocks: Sulfate-Free, Cruelty-Free, Professional Grade, Free Shipping $50+
2. **Hero Banner** → Upload hero image, set headline, subheading, CTA
3. **Category Cards** → Upload SKINCARE and HAIRCARE lifestyle images, set collection links
4. **Featured Products** → Add all 4 products, optionally set badge text (e.g., "Bestseller")

---

## Hair Type Quiz

The quiz popup (`ela-quiz.js`) automatically shows on first visit after **5 seconds**.  
It will **not** show again once the visitor answers or closes it (stored in `localStorage`).

To trigger it manually from a button anywhere:
```html
<button data-open-quiz>Take the Hair Quiz</button>
```

To recommend different products, edit the `getRecommendation()` function in `ela-quiz.js`.

---

## Free Shipping Progress Bar

The threshold is set to **$50** in `ela-cart-progress.js`:
```javascript
const ELA_SHIPPING_THRESHOLD = 5000; // cents
```
Change this value if your free shipping threshold is different.

---

## Mobile Sticky Add to Cart

On mobile, the "Add to Cart" button becomes sticky at the bottom of the screen  
when the main ATC button scrolls out of view. This is handled in `templates/product.ela.liquid`  
using the `IntersectionObserver` API — no additional configuration needed.

---

## Performance Notes

- All images use Shopify's CDN with responsive `srcset` for automatic format optimization
- Non-critical scripts loaded with `defer`
- Lazy-loading on all below-fold images (`loading="lazy"`)
- No autoplay video backgrounds
- Google Fonts loaded via a single `@import` in the CSS (consider inlining for production)

---

## Customization Reference

| What to change | Where |
|----------------|-------|
| Colors | `assets/ela-custom.css` – `:root` CSS variables |
| Fonts | `assets/ela-custom.css` – Google Fonts import + `--ela-font-*` variables |
| Free shipping threshold | `assets/ela-cart-progress.js` – `ELA_SHIPPING_THRESHOLD` constant |
| Quiz questions / logic | `assets/ela-quiz.js` – `questions` array + `getRecommendation()` |
| Navigation links | `layout/theme.liquid` – nav section |
| Footer links | `layout/theme.liquid` – footer section |
| Bundle content | `sections/bundles.liquid` + Theme Editor |
