import React, { useState, useRef } from 'react';
import YouTube from 'react-youtube';

function App() {
  const [videoId, setVideoId] = useState('');
  const [currentId, setCurrentId] = useState(null);
  const [inPlay, setInPlay] = useState(false);
  const [segments, setSegments] = useState([]);
  const playerRef = useRef(null);

  const extractVideoId = (url) => {
    const match = url.match(/(?:v=|youtu\.be\/)([^&]+)/);
    return match ? match[1] : null;
  };

  const handleLoad = () => {
    const id = extractVideoId(videoId);
    setCurrentId(id);
  };

  const toggleLabel = () => {
    const time = playerRef.current.getCurrentTime();
    if (!inPlay) {
      setSegments([...segments, { start: time }]);
    } else {
      const last = segments[segments.length - 1];
      last.end = time;
      setSegments([...segments.slice(0, -1), last]);
    }
    setInPlay(!inPlay);
  };

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(segments, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentId || 'segments'}.json`;
    a.click();
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Volleyball Labeler</h2>
      <input
        style={{ width: '300px' }}
        placeholder="Paste YouTube link"
        value={videoId}
        onChange={(e) => setVideoId(e.target.value)}
      />
      <button onClick={handleLoad}>Load Video</button>

      {currentId && (
        <>
          <YouTube
            videoId={currentId}
            onReady={(e) => (playerRef.current = e.target)}
            opts={{
      width: '100%',
      // height: '500', // or leave blank if using aspect ratio trick
    }}
          />
          <button onClick={toggleLabel} style={{ marginTop: 10 }}>
            {inPlay ? 'End In-Play' : 'Start In-Play'}
          </button>
          <button onClick={downloadJSON} style={{ marginLeft: 10 }}>
            Export Labels (JSON)
          </button>
        </>
      )}

      {segments.length > 0 && (
        <pre style={{ marginTop: 20 }}>
          {JSON.stringify(segments, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default App;
