// Create context menu when extension is installed
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzePokemonCard",
    title: "Analyze Pokemon Card",
    contexts: ["image"]
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzePokemonCard") {
    // Send message to content script to analyze the image
    chrome.tabs.sendMessage(tab.id, {
      action: "analyzeImage",
      imageUrl: info.srcUrl
    });
  }
});
