import React, { useEffect, useState } from "react";
import VideoFeed from "./components/VideoFeed";
import Map from "./components/Map";
import axios from "axios";
import "./App.css";

function App() {
  const [personCount, setPersonCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      axios
        .get("http://localhost:5000/person_count")
        .then((response) => {
          setPersonCount(response.data.person_count);
        })
        .catch((error) => {
          console.error("There was an error fetching the person count!", error);
        });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <h1>Person Detection and Mapping</h1>
      <VideoFeed />
      <Map personCount={personCount} />
    </div>
  );
}

export default App;
