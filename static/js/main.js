document.addEventListener("DOMContentLoaded", function () {
  var form = document.getElementById("search-form");
  var btn = document.getElementById("recommend-btn");
  var overlay = document.getElementById("loading-overlay");
  var loadingText = document.getElementById("loading-text");
  var searchBox = document.getElementById("query");

  // --- search submit: loading overlay with the sources being searched ---
  if (form) {
    form.addEventListener("submit", function (event) {
      var checked = form.querySelectorAll('input[name="sources"]:checked');
      if (!checked.length) {
        event.preventDefault();
        showToast("Select at least one source to search.");
        return;
      }
      if (btn) { btn.disabled = true; }
      if (overlay) {
        var names = Array.prototype.map.call(checked, function (c) {
          return c.closest(".toggle-chip").textContent.trim();
        });
        var i = 0;
        if (loadingText) {
          loadingText.textContent = "Searching " + names[0] + "…";
          setInterval(function () {
            i = (i + 1) % (names.length + 1);
            loadingText.textContent = i === names.length
              ? "Ranking papers by semantic similarity…"
              : "Searching " + names[i] + "…";
          }, 1100);
        }
        overlay.hidden = false;
      }
    });
  }

  // Grow the search box with its content (long queries stay fully visible)
  function autoResize() {
    searchBox.style.height = "auto";
    searchBox.style.height = Math.min(searchBox.scrollHeight, 220) + "px";
  }
  if (searchBox) {
    searchBox.addEventListener("input", autoResize);
    window.addEventListener("resize", autoResize);
    autoResize();
  }

  // Enter submits, Shift+Enter adds a new line
  if (searchBox && form) {
    searchBox.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (form.requestSubmit) { form.requestSubmit(); } else { form.submit(); }
      }
    });
  }

  // Coming back via the browser's back button: don't leave the overlay up
  window.addEventListener("pageshow", function () {
    if (overlay) { overlay.hidden = true; }
    if (btn) { btn.disabled = false; }
  });

  // --- example topic chips ---
  document.querySelectorAll(".example-chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      if (searchBox) {
        searchBox.value = chip.getAttribute("data-topic");
        autoResize();
        searchBox.focus();
      }
    });
  });

  // --- collapsible abstracts (only when actually longer than 3 lines) ---
  document.querySelectorAll("[data-collapsible]").forEach(function (box) {
    var text = box.querySelector(".abstract-text");
    var toggle = box.querySelector(".abstract-toggle");
    box.classList.add("is-collapsed");
    if (text.scrollHeight > text.clientHeight + 2) {
      toggle.hidden = false;
      toggle.addEventListener("click", function () {
        var collapsed = box.classList.toggle("is-collapsed");
        toggle.textContent = collapsed ? "Show full abstract" : "Show less";
      });
    } else {
      box.classList.remove("is-collapsed");
    }
  });

  // --- copy DOI link ---
  document.querySelectorAll(".copy-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      var value = b.getAttribute("data-copy");
      var done = function () { showToast("DOI link copied to clipboard"); };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(value).then(done, function () { fallbackCopy(value); done(); });
      } else {
        fallbackCopy(value); done();
      }
    });
  });

  function fallbackCopy(value) {
    var ta = document.createElement("textarea");
    ta.value = value;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  var toastTimer;
  function showToast(message) {
    var t = document.querySelector(".toast");
    if (!t) {
      t = document.createElement("div");
      t.className = "toast";
      t.setAttribute("role", "status");
      document.body.appendChild(t);
    }
    t.textContent = message;
    t.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove("is-visible"); }, 2200);
  }
});
