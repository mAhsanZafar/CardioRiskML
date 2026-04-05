/**
 * ELA Professional – Hair Type Quiz Popup
 *
 * Shows a 3-question popup on first visit (after 5 s).
 * Result stored in localStorage to prevent repeat display.
 * Recommends: Shampoo / Hair Mask / Hair Serum based on answers.
 */

(function () {
  'use strict';

  var STORAGE_KEY = 'ela_quiz_complete';

  /* -------- Quiz data -------- */
  var questions = [
    {
      id:      'hair_type',
      text:    'How would you describe your hair type?',
      options: [
        { value: 'fine',     label: 'Fine & Limp' },
        { value: 'thick',    label: 'Thick & Coarse' },
        { value: 'damaged',  label: 'Damaged & Brittle' },
        { value: 'normal',   label: 'Normal / Balanced' }
      ]
    },
    {
      id:      'concern',
      text:    'What is your main hair concern?',
      options: [
        { value: 'volume',   label: 'I want more volume & fullness' },
        { value: 'moisture', label: 'I need moisture & hydration' },
        { value: 'repair',   label: 'I want to repair & strengthen' },
        { value: 'scalp',    label: 'Scalp health & cleanliness' }
      ]
    },
    {
      id:      'routine',
      text:    'What does your current hair care routine look like?',
      options: [
        { value: 'none',     label: 'Nothing specific – just shampoo' },
        { value: 'some',     label: 'A few products here and there' },
        { value: 'full',     label: 'A dedicated routine already' }
      ]
    }
  ];

  /**
   * Determine recommended product based on answers.
   * @param {Object} answers – { hair_type, concern, routine }
   * @returns {{ title, reason, url, btnText }}
   */
  function getRecommendation(answers) {
    var concern   = answers.concern   || '';
    var hairType  = answers.hair_type || '';

    /* Priority: repair → Mask; volume/scalp → Serum; moisture/normal → Shampoo */
    if (hairType === 'damaged' || concern === 'repair') {
      return {
        title:   'Keratin Hair Mask',
        reason:  'Your hair needs deep reconstruction. Our Keratin Hair Mask — enriched with Brazilian Keratin and Oil Complex — will rebuild strength, restore elasticity, and repair damage.',
        url:     '/products/keratin-hair-mask',
        btnText: 'Shop Hair Mask'
      };
    }

    if (concern === 'volume' || hairType === 'fine') {
      return {
        title:   'Multi-Peptide Serum for Hair Density',
        reason:  'Our peptide-powered serum targets root density, making hair visibly thicker, fuller, and healthier from the very first application.',
        url:     '/products/multi-peptide-serum-hair-density',
        btnText: 'Shop Hair Serum'
      };
    }

    return {
      title:   'Keratin Sulfate-Free Shampoo',
      reason:  'A gentle, sulfate-free cleanse that preserves your hair's natural moisture while improving manageability for all hair types.',
      url:     '/products/keratin-sulfate-free-shampoo',
      btnText: 'Shop Shampoo'
    };
  }

  /* -------- DOM helpers -------- */
  function el(selector) {
    return document.querySelector(selector);
  }

  function buildQuizHTML() {
    var stepsHTML = questions.map(function (q, i) {
      var optHTML = q.options.map(function (opt) {
        return (
          '<div class="ela-quiz__option" data-question="' + q.id + '" data-value="' + escapeAttr(opt.value) + '">' +
          escapeHtml(opt.label) +
          '</div>'
        );
      }).join('');

      return (
        '<div class="ela-quiz__step' + (i === 0 ? ' is-active' : '') + '" data-step="' + i + '">' +
        '<p class="ela-quiz__question">' + escapeHtml(q.text) + '</p>' +
        '<div class="ela-quiz__options">' + optHTML + '</div>' +
        '</div>'
      );
    }).join('');

    var dotsHTML = questions.map(function (_, i) {
      return '<div class="ela-quiz__progress-dot' + (i === 0 ? ' is-active' : '') + '" data-dot="' + i + '"></div>';
    }).join('');

    return (
      '<div class="ela-quiz-overlay" id="ela-quiz-overlay" role="dialog" aria-modal="true" aria-label="Hair Type Quiz">' +
        '<div class="ela-quiz">' +
          '<button class="ela-quiz__close" aria-label="Close quiz">✕</button>' +
          '<p class="ela-quiz__eyebrow">Personalised Recommendation</p>' +
          '<h2 class="ela-quiz__title">Find Your Perfect Hair Routine</h2>' +
          '<p class="ela-quiz__subtitle">Answer 3 quick questions — we\'ll recommend the best ELA product for you.</p>' +
          '<div class="ela-quiz__progress">' + dotsHTML + '</div>' +
          stepsHTML +
          '<div class="ela-quiz__result" data-step="result">' +
            '<div class="ela-quiz__result-icon">✦</div>' +
            '<p class="ela-quiz__result-title" id="ela-quiz-result-title"></p>' +
            '<p class="ela-quiz__result-text" id="ela-quiz-result-text"></p>' +
            '<a class="ela-btn ela-quiz__result-btn" id="ela-quiz-result-btn" href="#">Shop Now</a>' +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function escapeAttr(str) {
    return String(str).replace(/"/g, '&quot;');
  }

  /* -------- Quiz controller -------- */
  function QuizController() {
    this.currentStep = 0;
    this.answers     = {};
    this.overlay     = null;
  }

  QuizController.prototype.init = function () {
    // Skip if already completed
    if (localStorage.getItem(STORAGE_KEY)) return;

    document.body.insertAdjacentHTML('beforeend', buildQuizHTML());
    this.overlay = el('#ela-quiz-overlay');

    this._bindEvents();

    // Show after 5 seconds on first visit
    setTimeout(function () {
      this.open();
    }.bind(this), 5000);
  };

  QuizController.prototype.open = function () {
    if (!this.overlay) return;
    this.overlay.classList.add('is-open');
    document.body.style.overflow = 'hidden';
    // Focus the first interactive element
    var firstOption = this.overlay.querySelector('.ela-quiz__option');
    if (firstOption) firstOption.focus();
  };

  QuizController.prototype.close = function () {
    if (!this.overlay) return;
    this.overlay.classList.remove('is-open');
    document.body.style.overflow = '';
    localStorage.setItem(STORAGE_KEY, '1');
  };

  QuizController.prototype.selectOption = function (optionEl) {
    var question = optionEl.dataset.question;
    var value    = optionEl.dataset.value;

    this.answers[question] = value;

    // Deselect siblings
    var siblings = optionEl.closest('.ela-quiz__options').querySelectorAll('.ela-quiz__option');
    siblings.forEach(function (s) { s.classList.remove('is-selected'); });
    optionEl.classList.add('is-selected');

    // Auto-advance after brief pause
    setTimeout(function () {
      this.advance();
    }.bind(this), 340);
  };

  QuizController.prototype.advance = function () {
    var totalSteps = questions.length;

    if (this.currentStep < totalSteps - 1) {
      // Move to next step
      var current = this.overlay.querySelector('[data-step="' + this.currentStep + '"]');
      if (current) current.classList.remove('is-active');

      this.currentStep += 1;

      var next = this.overlay.querySelector('[data-step="' + this.currentStep + '"]');
      if (next) next.classList.add('is-active');

      // Update progress dots
      this.overlay.querySelectorAll('.ela-quiz__progress-dot').forEach(function (dot, i) {
        dot.classList.toggle('is-active', i <= this.currentStep);
      }.bind(this));

    } else {
      // Show result
      var lastStep = this.overlay.querySelector('[data-step="' + this.currentStep + '"]');
      if (lastStep) lastStep.classList.remove('is-active');

      this.showResult();
    }
  };

  QuizController.prototype.showResult = function () {
    var rec = getRecommendation(this.answers);

    var titleEl  = el('#ela-quiz-result-title');
    var textEl   = el('#ela-quiz-result-text');
    var btnEl    = el('#ela-quiz-result-btn');
    var resultEl = this.overlay.querySelector('[data-step="result"]');

    if (titleEl)  titleEl.textContent  = 'We recommend: ' + rec.title;
    if (textEl)   textEl.textContent   = rec.reason;
    if (btnEl) {
      btnEl.textContent = rec.btnText;
      btnEl.href        = rec.url;
    }

    if (resultEl) resultEl.classList.add('is-active');

    // Mark all dots as active
    this.overlay.querySelectorAll('.ela-quiz__progress-dot').forEach(function (dot) {
      dot.classList.add('is-active');
    });

    localStorage.setItem(STORAGE_KEY, '1');
  };

  QuizController.prototype._bindEvents = function () {
    var self = this;

    this.overlay.addEventListener('click', function (e) {
      // Close button
      if (e.target.closest('.ela-quiz__close')) {
        self.close();
        return;
      }

      // Click outside the modal
      if (e.target === self.overlay) {
        self.close();
        return;
      }

      // Option selection
      var option = e.target.closest('.ela-quiz__option');
      if (option) {
        self.selectOption(option);
      }
    });

    // Keyboard support
    this.overlay.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') self.close();
    });
  };

  /* -------- Bootstrap -------- */
  document.addEventListener('DOMContentLoaded', function () {
    var quiz = new QuizController();
    quiz.init();

    // Allow manual trigger from other elements, e.g. a "Take the quiz" link
    document.addEventListener('click', function (e) {
      if (e.target.closest('[data-open-quiz]')) {
        e.preventDefault();
        quiz.open();
      }
    });

    window.ELAQuiz = quiz;
  });
})();
