import React, { Suspense } from "react";
import mobiscroll from "@mobiscroll/react";
import "@mobiscroll/react/dist/css/mobiscroll.min.css";
import { connect } from "react-redux";
import { submitPredict, getVisualAnalysis, clearAnalysis } from "../actions";
import { Select } from "../Select"

mobiscroll.settings = {
  theme: "ios",
  themeVariant: "light",
};

class TestPageComponent extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      options: [],
      hasURL: false,
      uploadedURL: "",
      showImage: false,
      analysisType: ''
    };

    this.handleSetURL = this.handleSetURL.bind(this);
    this.clearURL = this.clearURL.bind(this);
    this.handlePredict = this.handlePredict.bind(this);
    this.handleShowImage = this.handleShowImage.bind(this);
    this.handleGetAnalysis = this.handleGetAnalysis.bind(this);
    this.handleSetAnalysisType = this.handleSetAnalysisType.bind(this);
    this.handleClearAll = this.handleClearAll.bind(this);
  }

  async componentDidUpdate(prevProps) {
    if(prevProps.imageDelivered !== this.props.imageDelivered){
      if (this.props.imageDelivered) {
        setTimeout(this.handleShowImage, 1000);
      }
    }
    if (prevProps.clearAll !== this.props.clearAll) {
      if (this.props.clearAll) {
        this.handleClearAll()
      }
    }
  }

  handleShowImage(){
    this.setState({showImage: true})
  }

    handleClearAll(){
      this.setState({
        options: [],
        hasURL: false,
        uploadedURL: "",
        showImage: false,
        analysisType: ''
    })
      this.props.clearAll()
  }

  handleSetAnalysisType(e) {
    const idx = e.valueText.indexOf('--')
    const analysisType = e.valueText.slice(0, idx)
    this.setState({ analysisType })
  }

  handleSetURL(e) {
    this.setState({ hasURL: true, uploadedURL: e.target.value });
  }

  clearURL() {
    this.setState({ hasURL: false, uploadedURL: "" });
  }

  async handleGetAnalysis() {
    await this.props.getVisual();
  }

  async handlePredict() {
    const { uploadedURL, analysisType } = this.state;
    const { predict } = this.props;
    console.log('TestPage.js, 83 ==> start analysis');
    await predict(uploadedURL, analysisType);
    console.log('TestPage.js, 85 ==> step 2');
  }

  render() {
    const { hasURL, uploadedURL, showImage, analysisType } = this.state;
    const { currentPrediction, predAccuracy, imgStr, hasImg, clearAll} = this.props;
    const hasAnalysisSet = analysisType.length > 0;
    return (
      <div>
        <mobiscroll.Form className="mbsc-form-grid" theme="ios" themeVariant="light">
          <div className="mbsc-grid">
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-8 mbsc-align-center">
                <mobiscroll.Note color="primary">Enter the url of an image, then select submit for image
                  identification and prediction accuracy probability score.</mobiscroll.Note>
              </div>
            </div>
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-12 mbsc-col-md-6">
                <mobiscroll.Input
                  inputStyle="box"
                  labelStyle="floating"
                  value={uploadedURL}
                  placeholder="What is the url of the image you'd like to analyze?"
                  onChange={this.handleSetURL}
                >
                  Paste Image URL
                </mobiscroll.Input>              
              </div>
            </div>
            <Select onChange={this.handleSetAnalysisType} clearAll={clearAll}/>
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-sm-8">
                {hasURL && (
                  <div>
                    <h4>Uploaded Image</h4>
                    <img
                      alt={uploadedURL}
                      src={uploadedURL}
                      className="mbsc-col-sm-8"
                    />
                  </div>
                )}
              </div>
            </div>
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-6">
                <div className="mbsc-btn-group-block">
                  {hasAnalysisSet && (
                    <mobiscroll.Button
                    color="success"
                    onClick={this.handlePredict}
                  >
                    Submit Image
                  </mobiscroll.Button>)
                  }
                </div>
              </div>
            </div>
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-6">
                <div className="mbsc-btn-group-block">
                  <mobiscroll.Button
                    color="warning"
                    onClick={this.handleClearAll}
                  >
                    Clear State / Start Over
                  </mobiscroll.Button>
                </div>
              </div>
            </div>
            <div className="mbsc-row mbsc-justify-content-center">
              <div className="mbsc-col-sm-8">
                {hasURL &&
                  showImage && (
                    <div>
                      <h4>Classification/ Prediction Data </h4>
                      <h6>*will update in 1-2 seconds</h6>
                      <div className="mbsc-btn-group-block">
                        <mobiscroll.Note color="secondary">
                          <p className="mbsc-align-left">
                            {`Image classification: ${currentPrediction}`}
                          </p>
                          <p className="mbsc-align-left">
                            {`Prediction accuracy: ${predAccuracy}%`}
                          </p>
                        </mobiscroll.Note>
                      </div>
                    </div>
                  )}
              </div>
            </div>
            {hasURL &&
              showImage && hasImg && (
                <div className="mbsc-row mbsc-justify-content-center">
                  <div className="mbsc-col-12">
                    <div className="mbsc-btn-group-block">
                    <img className="mbsc-col-9" src={`data:image/jpeg;base64,${imgStr.img}`} alt="visual" />
                    </div>
                  </div>
                </div>
              )}
          </div>
        </mobiscroll.Form>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    imageDelivered: state.simpleReducer.imageDelivered,
    analysisType: state.simpleReducer.analysisType,
    clearAll: state.simpleReducer.clearAll
  };
};

const mapDispatchToProps = (dispatch) => ({
  predict: (url, type) => dispatch(submitPredict(url, type)),
  getVisual: () => dispatch(getVisualAnalysis()),
  clearAll: () => dispatch(clearAnalysis()),
});

export const TestPage = connect(
  mapStateToProps,
  mapDispatchToProps
)(TestPageComponent);
