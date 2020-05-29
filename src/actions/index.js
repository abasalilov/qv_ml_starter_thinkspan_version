const request = require("request");

const LOCAL = "/api/predict";

const options = {
  method: "POST",
  url: "",
  mode: "no-cors", // no-cors,
  headers: {
    "Content-Type": "application/x-www-form-urlencoded",
    Accept: "*/*",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-cache",
  },
};

export const PREDICT_ERR = "PREDICT_ERR";
export const CONFIRM_PREDICT = "CONFIRM_PREDICT";
export const SELECTED_TYPE = "SELECTED_TYPE";


export const submitPredict = (URL, type) => async (dispatch) => {
  const URLA = "http://167.172.125.42:5000/check?q=" + `${URL}&type=${type}`;
  const raw = "";
  var requestOptions = {
    method: "POST",
    body: raw,
    redirect: "follow",
    mode: "no-cors",
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  };
  // console.log("index.js, 59 ==> here in getting stuff");
  await fetch(URLA, requestOptions).catch((err) => {
    console.log("error", err);
  });

  setTimeout(
    dispatch({
      type: CONFIRM_PREDICT,
    }),
    1000
  );
};

export const selectAnalysis = (t) => async (dispatch) => {
  dispatch({
    type: SELECTED_TYPE,
    payload: t
  })
}

export const CLEAR_ALL = "CLEAR_ALL"

export const clearAnalysis = (t) => async (dispatch) => {
  const CLEAR_URL = "http://167.172.125.42:5000/clear";
  var requestOptions = {
    method: "GET",
    redirect: "follow",
    mode: "no-cors",
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  // }
  // console.log('index.js, 66 ==> clear');
  // await fetch(CLEAR_URL, requestOptions)
  //   .then(answer => console.log('index.js, 67 ==> answer', answer))
  //   .catch((err) => {
  //     console.log("error", err);
  //   });
  }
  dispatch({
    type: CLEAR_ALL
  })
}


export const getVisualAnalysis = (URL, analysisType) => async (dispatch) => {
  const URLA = `http://167.172.125.42:5000/image/${analysisType}`;
  var requestOptions = {
    method: "GET",
    redirect: "follow",
    mode: "no-cors",
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  };
  await fetch(URLA, requestOptions)
  .then(img => {
    console.log('index.js, 106 ==> img', img);
  }).catch(err => {
    console.log('index.js, 108 ==> err', err);
  })
};
