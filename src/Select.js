import React from 'react'
import mobiscroll from '@mobiscroll/react';
import '@mobiscroll/react/dist/css/mobiscroll.min.css';
import { connect } from "react-redux";
import { selectAnalysis } from "./actions";

mobiscroll.settings = {
    theme: 'ios',
    themeVariant: 'light'
};

const methods = [["Gradient", "Function"], ["SmoothGrad", "Function"], ["Deconvnet", "Signal"], ["Guided-Backprop", "Signal"], ["PatternNet", "Signal"], ["PatternAttribution", "Interaction"],
    ["DeepTaylor", "Interaction"], ["Input*Gradient", "Interaction"], ["Integrated-Gradients", "Interaction"], ["LRP-Z", "Interaction"], ["LRP-Epsilon", "Interaction"], ["LRP-PresetAFlat", "Interaction"], ["LRP-PresetBFlat", "Interaction"], ["All", "All"]]

class SelectComponent extends React.Component {

    getResposiveSetting = () => {
        return {
            small: {
                display: 'bubble'
            },
            medium: {
                touchUi: false
            }
        }
    }

    render() {
        const { onChange } = this.props
        return (
            <mobiscroll.Form>
                <div className="mbsc-grid">
                    <div className="mbsc-row mbsc-justify-content-center">
                        <div className="mbsc-col-sm-6">
                            <mobiscroll.FormGroup inset>
                                <mobiscroll.FormGroupTitle>Select Analysis Type</mobiscroll.FormGroupTitle>
                                <label>
                                    Analysis Types
                                    <mobiscroll.Select
                                        onChange={(e, i) => onChange(e)}
                                        responsive={this.getResposiveSetting()}
                                        value={1}
                                    >
                                        {methods.map((m,i) => <option key={i+Math.random()} value={m[0]}>{`${m[0]} -- ${m[1]}`}</option>)}
                                    </mobiscroll.Select>
                                </label>
                            </mobiscroll.FormGroup>
                        </div>
                    </div>
                </div>
            </mobiscroll.Form>
        );
    }
}

const mapStateToProps = (state) => {
    return {
        imageDelivered: state.simpleReducer.imageDelivered
    };
};

const mapDispatchToProps = (dispatch) => ({
    selectAnalysisType: (t) => dispatch(selectAnalysis(t)),
});

export const Select = connect(
    mapStateToProps,
    mapDispatchToProps
)(SelectComponent);


