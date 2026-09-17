/**
 * YouTube IFrame Player API Controller
 */

let playerInstance = null;
let isApiReady = false;
let pendingVideoId = null;

// Initialize YouTube IFrame API
export function initYouTubeApi() {
  if (window.YT && window.YT.Player) {
    isApiReady = true;
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    window.onYouTubeIframeAPIReady = () => {
      isApiReady = true;
      if (pendingVideoId) {
        loadVideo(pendingVideoId);
      }
      resolve();
    };

    if (!document.getElementById("yt-iframe-api-script")) {
      const tag = document.createElement("script");
      tag.id = "yt-iframe-api-script";
      tag.src = "https://www.youtube.com/iframe_api";
      const firstScriptTag = document.getElementsByTagName("script")[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }
  });
}

export function loadVideo(videoId) {
  if (!isApiReady || !window.YT || !window.YT.Player) {
    pendingVideoId = videoId;
    return;
  }

  const placeholder = document.getElementById("player-placeholder");
  if (placeholder) {
    placeholder.style.display = "none";
  }

  if (playerInstance && typeof playerInstance.loadVideoById === "function") {
    playerInstance.loadVideoById(videoId);
  } else {
    playerInstance = new window.YT.Player("youtube-player", {
      height: "100%",
      width: "100%",
      videoId: videoId,
      playerVars: {
        playsinline: 1,
        rel: 0,
        modestbranding: 1,
      },
      events: {
        onReady: () => {
          console.log("YouTube Player is ready");
        },
      },
    });
  }
}

export function seekTo(seconds) {
  if (playerInstance && typeof playerInstance.seekTo === "function") {
    playerInstance.seekTo(seconds, true);
    if (typeof playerInstance.playVideo === "function") {
      playerInstance.playVideo();
    }
  } else {
    console.warn("Player not ready to seek to", seconds);
  }
}

export function resetPlayer() {
  const placeholder = document.getElementById("player-placeholder");
  if (placeholder) {
    placeholder.style.display = "flex";
  }
  if (playerInstance && typeof playerInstance.stopVideo === "function") {
    playerInstance.stopVideo();
  }
}
