/**
 * ELA Professional – Cart Shipping Progress Bar
 * Calculates remaining amount for free shipping threshold ($50 default).
 * Threshold is read from the data-threshold attribute on .ela-shipping-bar,
 * which is set in layout/theme.liquid from theme settings.
 */

(function () {
  'use strict';

  var DEFAULT_THRESHOLD = 5000; // $50.00 in cents (fallback)

  function getThreshold() {
    var bar = document.querySelector('.ela-shipping-bar');
    var val = bar && parseInt(bar.dataset.threshold, 10);
    return (val && val > 0) ? val : DEFAULT_THRESHOLD;
  }

  /**
   * Formats a price in cents to a display string, e.g. "$12.50"
   * Relies on Shopify's window.Shopify.currency or falls back to USD.
   */
  function formatMoney(cents) {
    const dollars = (cents / 100).toFixed(2);
    return '$' + dollars;
  }

  /**
   * Renders the shipping progress bar state inside the cart drawer.
   * @param {number} cartTotal – cart total in cents
   */
  function renderShippingBar(cartTotal) {
    var bar = document.querySelector('.ela-shipping-bar');
    if (!bar) return;

    var fill    = bar.querySelector('.ela-shipping-bar__fill');
    var message = bar.querySelector('.ela-shipping-bar__message');
    if (!fill || !message) return;

    var ELA_SHIPPING_THRESHOLD = getThreshold();
    var remaining = ELA_SHIPPING_THRESHOLD - cartTotal;

    if (remaining <= 0) {
      fill.style.width = '100%';
      message.innerHTML = '🎉 You\'ve unlocked <strong>free shipping!</strong>';
    } else {
      const pct = Math.min((cartTotal / ELA_SHIPPING_THRESHOLD) * 100, 100);
      fill.style.width = pct + '%';
      message.innerHTML =
        'Add <strong>' + formatMoney(remaining) + '</strong> more for free shipping';
    }
  }

  /**
   * Fetches the current Shopify cart JSON and updates the progress bar.
   */
  function updateShippingBar() {
    fetch('/cart.js', { headers: { 'Content-Type': 'application/json' } })
      .then(function (res) { return res.json(); })
      .then(function (cart) {
        renderShippingBar(cart.total_price);
      })
      .catch(function (err) {
        console.warn('[ELA] Could not fetch cart:', err);
      });
  }

  /**
   * Opens the cart drawer.
   */
  function openCartDrawer() {
    const drawer  = document.querySelector('.ela-cart-drawer');
    const overlay = document.querySelector('.ela-cart-overlay');
    if (drawer)  drawer.classList.add('is-open');
    if (overlay) overlay.classList.add('is-open');
    document.body.style.overflow = 'hidden';
    updateShippingBar();
    renderCartItems();
  }

  /**
   * Closes the cart drawer.
   */
  function closeCartDrawer() {
    const drawer  = document.querySelector('.ela-cart-drawer');
    const overlay = document.querySelector('.ela-cart-overlay');
    if (drawer)  drawer.classList.remove('is-open');
    if (overlay) overlay.classList.remove('is-open');
    document.body.style.overflow = '';
  }

  /**
   * Renders cart line items inside the drawer body.
   */
  function renderCartItems() {
    const body = document.querySelector('.ela-cart-drawer__body');
    const subtotalEl = document.querySelector('.ela-cart-drawer__subtotal-value');
    if (!body) return;

    fetch('/cart.js', { headers: { 'Content-Type': 'application/json' } })
      .then(function (res) { return res.json(); })
      .then(function (cart) {
        if (subtotalEl) {
          subtotalEl.textContent = formatMoney(cart.total_price);
        }

        // Clear existing items (keep upsell block if present)
        const upsell = body.querySelector('.ela-upsell');
        body.innerHTML = '';
        if (upsell) body.appendChild(upsell);

        if (cart.item_count === 0) {
          body.insertAdjacentHTML('afterbegin',
            '<p style="text-align:center;padding:40px 0;color:#999;font-size:13px;">Your cart is empty.</p>'
          );
          toggleUpsell(cart, body);
          return;
        }

        const fragment = document.createDocumentFragment();

        cart.items.forEach(function (item) {
          const div = document.createElement('div');
          div.className = 'ela-cart-item';
          div.dataset.key = item.key;
          div.innerHTML =
            '<img class="ela-cart-item__image" src="' + item.image + '" alt="' + escapeHtml(item.product_title) + '" loading="lazy">' +
            '<div class="ela-cart-item__info">' +
              '<div class="ela-cart-item__title">' + escapeHtml(item.product_title) + '</div>' +
              '<div class="ela-cart-item__price">' + formatMoney(item.line_price) + '</div>' +
              '<div class="ela-cart-item__qty">' +
                '<button class="ela-cart-item__qty-btn" data-action="decrease" data-key="' + item.key + '" aria-label="Decrease quantity">−</button>' +
                '<span class="ela-cart-item__qty-num">' + item.quantity + '</span>' +
                '<button class="ela-cart-item__qty-btn" data-action="increase" data-key="' + item.key + '" aria-label="Increase quantity">+</button>' +
              '</div>' +
            '</div>';
          fragment.appendChild(div);
        });

        body.insertBefore(fragment, body.firstChild);
        toggleUpsell(cart, body);
      })
      .catch(function (err) {
        console.warn('[ELA] Could not render cart items:', err);
      });
  }

  /**
   * Show/hide the bundle upsell based on cart contents.
   * Upsell appears when the Hair Repair Bundle is not already in cart
   * but a qualifying hair product is present.
   */
  function toggleUpsell(cart, body) {
    const upsell = body.querySelector('.ela-upsell');
    if (!upsell) return;

    const hairTags = ['shampoo', 'hair-mask', 'hair-serum'];
    const hasBundleItem = cart.items.some(function (item) {
      const t = (item.product_type || '').toLowerCase();
      return hairTags.some(function (tag) { return t.indexOf(tag) !== -1; });
    });
    const hasBundleProduct = cart.items.some(function (item) {
      return (item.product_title || '').toLowerCase().indexOf('bundle') !== -1;
    });

    upsell.style.display = (hasBundleItem && !hasBundleProduct) ? 'block' : 'none';
  }

  /**
   * Updates a cart line item quantity via AJAX.
   * @param {string} key – line item key
   * @param {number} quantity – new quantity (0 = remove)
   */
  function updateCartItem(key, quantity) {
    fetch('/cart/change.js', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: key, quantity: quantity })
    })
      .then(function (res) { return res.json(); })
      .then(function (cart) {
        renderCartItems();
        updateShippingBar();
        updateCartCount(cart.item_count);
      })
      .catch(function (err) {
        console.warn('[ELA] Could not update cart item:', err);
      });
  }

  /**
   * Updates the header cart count badge.
   */
  function updateCartCount(count) {
    document.querySelectorAll('[data-cart-count]').forEach(function (el) {
      el.textContent = count;
      el.style.display = count > 0 ? 'inline' : 'none';
    });
  }

  /**
   * Handles "Add to Cart" form submissions across the page.
   */
  function handleAddToCart(form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const formData = new FormData(form);

      fetch('/cart/add.js', {
        method: 'POST',
        body: formData
      })
        .then(function (res) { return res.json(); })
        .then(function () {
          openCartDrawer();
        })
        .catch(function (err) {
          console.warn('[ELA] Add to cart failed:', err);
        });
    });
  }

  /**
   * Minimal HTML escape to avoid XSS when inserting product titles.
   */
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ---- Init ---- */
  document.addEventListener('DOMContentLoaded', function () {
    // Cart triggers
    document.addEventListener('click', function (e) {
      if (e.target.closest('[data-open-cart]')) {
        e.preventDefault();
        openCartDrawer();
      }

      if (e.target.closest('.ela-cart-drawer__close') ||
          e.target.closest('.ela-cart-overlay')) {
        closeCartDrawer();
      }

      // Qty buttons
      const qtyBtn = e.target.closest('.ela-cart-item__qty-btn');
      if (qtyBtn) {
        const key = qtyBtn.dataset.key;
        const item = qtyBtn.closest('.ela-cart-item');
        const numEl = item ? item.querySelector('.ela-cart-item__qty-num') : null;
        let qty = numEl ? parseInt(numEl.textContent, 10) : 1;
        if (qtyBtn.dataset.action === 'increase') qty += 1;
        if (qtyBtn.dataset.action === 'decrease') qty -= 1;
        updateCartItem(key, Math.max(0, qty));
      }
    });

    // Add to cart forms
    document.querySelectorAll('form[action="/cart/add"]').forEach(handleAddToCart);

    // Keyboard: close drawer on Escape
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeCartDrawer();
    });

    // Initial count update
    fetch('/cart.js', { headers: { 'Content-Type': 'application/json' } })
      .then(function (res) { return res.json(); })
      .then(function (cart) { updateCartCount(cart.item_count); })
      .catch(function () {});
  });

  // Expose openCartDrawer globally for theme.liquid cart icon usage
  window.ELACart = { open: openCartDrawer, close: closeCartDrawer };
})();
