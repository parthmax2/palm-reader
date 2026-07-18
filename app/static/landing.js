const uploadCard = document.querySelector("#uploadCard");
const palmInput = document.querySelector("#palmImage");
const cameraInput = document.querySelector("#cameraImage");
const beginButton = document.querySelector("#beginButton");
const cameraButton = document.querySelector("#cameraButton");
const previewBg = document.querySelector("#previewBg");
const uploadHint = document.querySelector("#uploadHint");
const statusStrip = document.querySelector("#statusStrip");
const readingExperience = document.querySelector("#readingExperience");
const palmPreview = document.querySelector("#palmPreview");
const scanSteps = [...document.querySelectorAll("#scanSteps span")];
const lakshanGrid = document.querySelector("#lakshanGrid");
const clarityRing = document.querySelector("#clarityRing");
const clarityValue = document.querySelector("#clarityValue");
const readingTitle = document.querySelector("#readingTitle");
const readingOverview = document.querySelector("#readingOverview");
const insightKicker = document.querySelector("#insightKicker");
const insightTitle = document.querySelector("#insightTitle");
const insightBody = document.querySelector("#insightBody");
const traitRow = document.querySelector("#traitRow");
const scanAnotherButton = document.querySelector("#scanAnotherButton");
const copySummaryButton = document.querySelector("#copySummaryButton");
const tabButtons = [...document.querySelectorAll(".tab-button")];
const langButtons = [...document.querySelectorAll(".lang-button")];

const SECTION_META = {
  personality: {
    title: { en: "Swabhav / Personality", hi: "स्वभाव / Personality" },
    kicker: { en: "Mastishka insight", hi: "मस्तिष्क संकेत" },
  },
  relationships: {
    title: { en: "Prem / Relationships", hi: "प्रेम / Relationships" },
    kicker: { en: "Hridaya insight", hi: "हृदय संकेत" },
  },
  career: {
    title: { en: "Karma / Direction", hi: "कर्म / दिशा" },
    kicker: { en: "Bhagya insight", hi: "भाग्य संकेत" },
  },
  health: {
    title: { en: "Jeevan Shakti / Vitality", hi: "जीवन-शक्ति / Vitality" },
    kicker: { en: "Jeevan insight", hi: "जीवन संकेत" },
  },
  signs: {
    title: { en: "Mukhya Lakshan / Key Signs", hi: "मुख्य लक्षण / Key Signs" },
    kicker: { en: "Palm map summary", hi: "हस्तरेखा सार" },
  },
};

const LINE_LABELS = {
  heart: { en: "Hridaya Rekha", hi: "हृदय रेखा" },
  head: { en: "Mastishka Rekha", hi: "मस्तिष्क रेखा" },
  life: { en: "Jeevan Rekha", hi: "जीवन रेखा" },
  fate: { en: "Bhagya Rekha", hi: "भाग्य रेखा" },
};

let previewUrl = null;
let selectedFile = null;
let latestReading = null;
let currentLang = "en";
let currentSection = "personality";
let isScanning = false;

function updateStatus(message) {
  statusStrip.innerHTML = `<span class="status-dot"></span>${message}`;
}

function setPreview(file) {
  if (!file) return;
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewBg.style.backgroundImage = `url("${previewUrl}")`;
  previewBg.classList.add("is-visible");
  palmPreview.src = previewUrl;
  uploadHint.textContent = file.name || "Palm image selected.";
  beginButton.textContent = "Reveal Palm Map";
  updateStatus("Palm image ready. Begin the scan when you are ready.");
}

function chooseFile(input) {
  input.value = "";
  input.click();
}

function setLanguage(lang) {
  currentLang = lang;
  langButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.lang === lang);
  });
  if (latestReading) renderReading();
}

function setActiveSection(section) {
  currentSection = section;
  tabButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.section === section);
  });
  renderInsight();
}

function animateScan() {
  scanSteps.forEach((step) => step.classList.remove("is-active", "is-done"));
  readingExperience.classList.remove("is-complete");

  scanSteps.forEach((step, index) => {
    window.setTimeout(() => {
      scanSteps.forEach((item, itemIndex) => {
        item.classList.toggle("is-done", itemIndex < index);
        item.classList.toggle("is-active", itemIndex === index);
      });
    }, index * 620);
  });
}

function scrollToExperience() {
  const top = readingExperience.getBoundingClientRect().top + window.scrollY - 28;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

async function requestReading() {
  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("options", JSON.stringify({ languages: ["en", "hi"], detail: "standard" }));

  const response = await fetch("/v1/readings", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Palm reading failed.");
  }
  return response.json();
}

async function beginScan() {
  if (!selectedFile) {
    chooseFile(palmInput);
    return;
  }
  if (isScanning) return;

  isScanning = true;
  beginButton.disabled = true;
  beginButton.textContent = "Tracing Lines...";
  updateStatus("Locating palm geometry and tracing the principal Rekha.");
  readingExperience.classList.add("is-visible");
  scrollToExperience();
  animateScan();

  try {
    const [reading] = await Promise.all([
      requestReading(),
      new Promise((resolve) => window.setTimeout(resolve, scanSteps.length * 620 + 500)),
    ]);
    latestReading = reading;
    readingExperience.classList.add("is-complete");
    updateStatus("Palm map ready. Explore your reading below.");
    renderReading();
  } catch (error) {
    updateStatus("Could not complete the scan. Try a clearer palm image.");
    insightTitle.textContent = "Scan could not be completed";
    insightBody.textContent = error.message;
  } finally {
    isScanning = false;
    beginButton.disabled = false;
    beginButton.textContent = "Reveal Palm Map";
  }
}

function featureText(name, feature) {
  if (!feature?.present) return `${LINE_LABELS[name][currentLang]}: not clearly visible`;
  const parts = [];
  if (feature.length) parts.push(feature.length);
  if (feature.curved !== null && feature.curved !== undefined) {
    parts.push(feature.curved ? "curved" : "straight");
  }
  if (feature.fork_end) parts.push("forked");
  const confidence = Math.round((feature.confidence || 0) * 100);
  return `${LINE_LABELS[name][currentLang]}: ${parts.join(" + ") || "detected"} • ${confidence}%`;
}

function renderLakshan(features) {
  const names = ["heart", "head", "life", "fate"];
  lakshanGrid.innerHTML = names
    .map((name) => {
      const feature = features[name];
      const label = LINE_LABELS[name][currentLang];
      const text = featureText(name, feature).replace(`${label}: `, "");
      return `<div class="lakshan-chip"><strong>${label}</strong><span>${text}</span></div>`;
    })
    .join("");
}

function renderTraits(reading) {
  const featureTraits = [];
  if (reading.features?.head?.present) featureTraits.push(currentLang === "hi" ? "विवेक" : "Vivek");
  if (reading.features?.life?.present) featureTraits.push(currentLang === "hi" ? "आत्मबल" : "Atmabal");
  if (!reading.features?.fate?.present || reading.features?.fate?.present) {
    featureTraits.push(currentLang === "hi" ? "कर्म" : "Karma");
  }
  traitRow.innerHTML = featureTraits.slice(0, 3).map((trait) => `<span>${trait}</span>`).join("");
}

function renderReading() {
  const localized = latestReading.readings?.[currentLang] || latestReading.readings?.en;
  const clarity = Math.round((latestReading.overall_confidence || 0) * 100);
  clarityRing.style.setProperty("--clarity", clarity);
  clarityValue.textContent = `${clarity}%`;
  readingTitle.textContent = currentLang === "hi" ? "आपका Palm Map तैयार है" : "Your Palm Map is Ready";
  readingOverview.textContent = localized?.overview || "";
  renderLakshan(latestReading.features || {});
  renderTraits(latestReading);
  setActiveSection(currentSection);
}

function renderInsight() {
  if (!latestReading) return;
  const localized = latestReading.readings?.[currentLang] || latestReading.readings?.en;
  const meta = SECTION_META[currentSection];
  insightKicker.textContent = meta.kicker[currentLang] || meta.kicker.en;
  insightTitle.textContent = meta.title[currentLang] || meta.title.en;
  insightBody.textContent = localized?.[currentSection] || "No reading available for this section.";
}

async function copySummary() {
  if (!latestReading) return;
  const localized = latestReading.readings?.[currentLang] || latestReading.readings?.en;
  const summary = [
    "Palmistry AI",
    localized.overview,
    localized.signs,
    latestReading.disclaimer?.[currentLang],
  ].filter(Boolean).join("\n\n");
  await navigator.clipboard.writeText(summary);
  updateStatus("Summary copied to clipboard.");
}

beginButton.addEventListener("click", beginScan);
cameraButton.addEventListener("click", () => chooseFile(cameraInput));

palmInput.addEventListener("change", (event) => setPreview(event.target.files?.[0]));
cameraInput.addEventListener("change", (event) => setPreview(event.target.files?.[0]));

langButtons.forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.lang));
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveSection(button.dataset.section));
});

scanAnotherButton.addEventListener("click", () => {
  latestReading = null;
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewBg.style.backgroundImage = "";
  previewBg.classList.remove("is-visible");
  palmPreview.removeAttribute("src");
  palmInput.value = "";
  cameraInput.value = "";
  beginButton.textContent = "Begin Palm Reading";
  uploadHint.textContent = "Drag & drop your palm image, or choose a photo.";
  updateStatus("Ready for a palm image");
  readingExperience.classList.remove("is-visible", "is-complete");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

copySummaryButton.addEventListener("click", copySummary);

["dragenter", "dragover"].forEach((eventName) => {
  uploadCard.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadCard.classList.add("is-dragging");
    updateStatus("Drop your palm image to begin.");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadCard.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadCard.classList.remove("is-dragging");
  });
});

uploadCard.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  setPreview(file);
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
