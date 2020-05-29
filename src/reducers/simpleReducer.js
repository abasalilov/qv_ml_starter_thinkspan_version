/*
 src/reducers/simpleReducer.js
*/
import { CONFIRM_PREDICT, PREDICT_ERR, SELECTED_TYPE , CLEAR_ALL} from '../actions';

export default (state = {imageDelivered: false, clearAll: true}, action) => {
  switch (action.type) {
    case CONFIRM_PREDICT:
      console.log('simpleReducer.js, 9 ==> action', action);
      return {
        imageDelivered: true,
        clearAll: false
      };
    case PREDICT_ERR:
      return {
        result: action.err,
      };
    case CLEAR_ALL:
      return {
        clearAll: true,
        imageDelivered: false,
      };
    case SELECTED_TYPE:
      return {
        analysisType: action.payload
      }
    default:
      return state;
  }
};
