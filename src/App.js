import React, { useState, useEffect } from "react";
import { connect } from "react-redux";
import "./App.css";
import { TestPage } from "./Pages/TestPage";

function App(props) {
  const [hasImg, setHasImg] = useState(0);
  const [imgStr, setImgStr] = useState(1);
  const [currentPrediction, setCurrentPrediction] = useState(2);
  const [predAccuracy, setPredAccuracy] = useState(3);
  useEffect(() => {
    const { imgDelivered } = props;

    if (imgDelivered) {
      const interval = setInterval(() => {
        fetch("http://167.172.125.42:5000/predict")
          .then((res) => res.json())
          .then((data) => {
            let predPercentage;
            const hasDeci = data[0][1].indexOf(".");
            predPercentage =
              hasDeci !== -1 ? data[0][1].slice(hasDeci + 1) : data[0][1];
            const pred = data[0][2];
            if (data[0][1] === "1.00") {
              predPercentage = 100;
            }
            setPredAccuracy(predPercentage);
            setCurrentPrediction(pred);
          })
          .catch((err) => console.log("App.js, 28 ==> err", err));
      }, 1000);
      setTimeout(() => clearInterval(interval), 2500);
    }
    if (imgDelivered) {
      const interval = setInterval(() => {
        fetch(`http://167.172.125.42:5000/image`)
          .then((res) => res.json())
          .then((imgData) => {
            setImgStr(imgData)
            setHasImg(true)
          })
          .catch((err) => console.log("App.js, 28 ==> err", err));
      }, 4000);
      setTimeout(() => clearInterval(interval), 400000);
    }
  }, [props]);
  return (
    <div className="App">
        <TestPage
          hasImg={hasImg}
          imgStr={imgStr}
          predAccuracy={predAccuracy}
          currentPrediction={currentPrediction}
          makeCallToPrediction={setCurrentPrediction}
        />
    </div>
  );
}

const mapDispatchToProps = {};

const mapStateToProps = (state) => {
  const { simpleReducer } = state;
  return {
    imgDelivered: simpleReducer.imageDelivered,
    analysisType: simpleReducer.analysisType
  };
};

const AppComponent = connect(mapStateToProps, mapDispatchToProps)(App);

export default AppComponent;
