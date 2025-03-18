import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import { DatasetProvider } from "./components/DatasetContext";

function App() {
  return (
    <DatasetProvider>
    <Router>
      <Routes>
        <Route path="/*" element={<Dashboard />} />
      </Routes>
    </Router>
    </DatasetProvider>
  );
}

export default App;
