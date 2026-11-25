// Theme: runs in <head> so the chosen theme applies before first paint.
// No stored choice -> follow the OS. The toggle stores "light" or "dark".
(function () {
  "use strict";
  var KEY = "tp-theme";
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) { /* storage disabled */ }
  if (stored === "light" || stored === "dark") root.setAttribute("data-theme", stored);

  function current() {
    var set = root.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("theme-toggle");
    if (!button) return;
    var label = function () {
      var next = current() === "dark" ? "light" : "dark";
      button.setAttribute("aria-label", "Switch to " + next + " theme");
      button.title = "Switch to " + next + " theme";
    };
    label();
    button.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (e) { /* storage disabled */ }
      label();
      document.dispatchEvent(new CustomEvent("themechange", { detail: next }));
    });
  });
})();
