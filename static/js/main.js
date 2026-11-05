/* LeafGuard AI — small, dependency-free interface helpers. */
(function () {
  "use strict";

  // Mobile navigation
  const navToggle = document.querySelector("[data-nav-toggle]");
  const siteNav = document.querySelector("[data-site-nav]");
  if (navToggle && siteNav) {
    navToggle.addEventListener("click", function () {
      const isOpen = siteNav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
      navToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
    });
    siteNav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        siteNav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  // File upload preview
  const form = document.querySelector("[data-detect-form]");
  const fileInput = document.querySelector("[data-file-input]");
  const uploadArea = document.querySelector("[data-upload-area]");
  const uploadLabel = document.querySelector("[data-upload-label]");
  const preview = document.querySelector("[data-upload-preview]");
  const previewImage = document.querySelector("[data-preview-image]");
  const previewName = document.querySelector("[data-preview-name]");
  const previewDetails = document.querySelector("[data-preview-details]");
  const clearButton = document.querySelector("[data-clear-file]");
  const uploadHint = document.querySelector("[data-upload-hint]");
  let objectUrl = null;

  const MAX_BYTES = 8 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function extensionOf(name) {
    const dot = name.lastIndexOf(".");
    return dot >= 0 ? name.slice(dot).toLowerCase() : "";
  }

  function resetPreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    if (previewImage) previewImage.removeAttribute("src");
    if (preview) preview.hidden = true;
    if (uploadLabel) uploadLabel.hidden = false;
    if (fileInput) fileInput.value = "";
    if (uploadHint) uploadHint.textContent = "Make sure the leaf is in focus and well lit.";
    uploadArea.classList.remove("is-dragover");
  }

  function showFile(file) {
    if (!file) return;
    const extension = extensionOf(file.name);
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      resetPreview();
      if (uploadHint) uploadHint.textContent = "Please choose a JPG, PNG or WEBP image.";
      return;
    }
    if (file.size > MAX_BYTES) {
      resetPreview();
      if (uploadHint) uploadHint.textContent = "That image is over 8 MB. Please choose a smaller file.";
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImage.src = objectUrl;
    previewImage.alt = "Preview of " + file.name;
    previewName.textContent = file.name;
    previewDetails.textContent = formatBytes(file.size) + " · ready to analyse";
    preview.hidden = false;
    uploadLabel.hidden = true;
  }

  if (fileInput && uploadArea) {
    fileInput.addEventListener("change", function () {
      showFile(fileInput.files && fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (eventName) {
      uploadArea.addEventListener(eventName, function (event) {
        event.preventDefault();
        uploadArea.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      uploadArea.addEventListener(eventName, function (event) {
        event.preventDefault();
        uploadArea.classList.remove("is-dragover");
      });
    });
    uploadArea.addEventListener("drop", function (event) {
      if (event.dataTransfer && event.dataTransfer.files.length) {
        const file = event.dataTransfer.files[0];
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        showFile(file);
      }
    });

    if (clearButton) {
      clearButton.addEventListener("click", resetPreview);
    }
  }

  // Show a loading state only after the browser has accepted the form.
  if (form) {
    form.addEventListener("submit", function () {
      const button = form.querySelector("[data-submit-button]");
      const label = form.querySelector("[data-submit-label]");
      if (button && !button.disabled) {
        button.disabled = true;
        if (label) label.textContent = "Analysing…";
      }
    });
  }
})();
