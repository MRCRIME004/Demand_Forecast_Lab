const $=id=>document.getElementById(id);
const MAX_FILE_BYTES=50_000_000;
const MAX_OBSERVATIONS=10_000;
const BROWSER_MAX_OBSERVATIONS=2_500;
const LARGE_FILE_WARNING_ROWS=500;
const MAX_PREVIEW_BARS=16;
const RECOMMENDED_GRAPH_KEYS=new Set(['01_demand','02_evaluation','03_future','14_cv_folds','15_dm_test','17_stl_decomposition','18_stationarity']);
const METHOD_BUTTONS=[
  ['SMA','Moving Average'],
  ['SES','Exponential Smoothing'],
  ['Weekday_DR','Weekday Regression'],
  ['Weekday_Naive','Weekday Naive'],
  ['Holt',"Holt's Linear Trend"],
  ['Weekday_Ridge','Ridge Regression'],
  ['Weekday_RF','Random Forest'],
  ['Holt_Winters','Holt-Winters'],
  ['SARIMA','SARIMA'],
  ['Ensemble','Top-3 Ensemble'],
  ['Comparison','Comparison']
];
const DEFAULT_UNITS={
  'Orders':'orders',
  'Sales':'units',
  'Production quantity':'units',
  'Electricity or energy use':'kWh',
  'Website traffic':'visits',
  'Custom':'units'
};
const INTEGER_COLUMNS=new Set([
  'Observation','Day_of_week','Validation_Rank','Rank','Fold','Train_End_Observation',
  'Fold_Start_Observation','Fold_End_Observation','Fold_Size','Lag','Window','Origin',
  'Horizon','Step','Forecast_Step','Weekday_Code','Backtest_Error_Count','Operating_Week',
  'Operating_Days','Count','Bin','Test_N','Lags_Used','Feature_Count',
  'Validation_Calibration_Errors','Folds','Folds_Won','Orders','Receipts',
  'Days_With_Unfilled','Completed_Cycles','Successful_Cycles','Order_Number',
  'Operating_Day','EOQ_Batch_Count','Previous_Observation','Current_Observation',
  'Minimum_Skipped_Weekdays','N','DF','sMAPE_zero_excluded','Ridge_Importance_Rank',
  'RF_Permutation_Rank','n_estimators','max_depth','min_samples_leaf','random_state'
]);
const FOUR_DEC_COLUMNS=new Set([
  'p_value','LB5_p','BP_p','ACF','PACF','CI_95_Lower','CI_95_Upper','Statistic',
  'Critical_5pct','R2','Adjusted_R2','Weight','RF_Impurity_Importance','ridge_lambda'
]);
const THREE_DEC_COLUMNS=new Set([
  'MASE','Mean_MASE','SD_MASE','Validation_MASE','Test_MASE','DM_HLN_Stat','Series_Skewness','Series_Kurtosis','t_stat'
]);
const COLUMN_DEFINITIONS={
  Validation_Rank:'Rank from 1 (best) to 10 based on validation Mean Absolute Error (MAE).',
  Model:'Internal identifier of the forecasting model.',
  Model_Label:'Full readable name of the forecasting method.',
  Role:'Indicates whether the model is the Selected Winner, a Top-3 Ensemble Member, or a Candidate.',
  MAE:'Mean Absolute Error — average daily miss in your unit (lower is better).',
  Validation_MAE:'Mean Absolute Error on the validation split (used to pick the winning model).',
  Test_MAE:'Mean Absolute Error on the untouched final test split.',
  Mean_MAE:'Average MAE across all expanding-window cross-validation folds.',
  RMSE:'Root Mean Squared Error — penalizes large single-day misses more heavily than MAE.',
  WAPE_pct:'Weighted Absolute Percentage Error — total absolute error divided by total actual demand (%).',
  MASE:'Mean Absolute Scaled Error — below 1.00 means the model beats the 1-step naive benchmark.',
  sMAPE_pct:'Symmetric Mean Absolute Percentage Error (0% to 200% scale, lower is better).',
  Bias:'Average (Forecast minus Actual) — positive means over-forecasting, negative means under-forecasting.',
  Tracking_Signal:'Cumulative bias divided by MAE — values beyond ±4 indicate systematic drift.',
  DM_HLN_Stat:'Harvey–Leybourne–Newbold Diebold–Mariano statistic (negative values beat the benchmark).',
  p_value:'Statistical significance p-value (values below 0.05 are statistically significant at the 5% level).',
  EOQ:'Economic Order Quantity — optimal batch size balancing ordering and holding costs.',
  safety_stock:'Buffer stock held against lead-time forecast uncertainty (0 when service level is omitted).',
  reorder_point:'Stock level (ROP) that triggers a new replenishment order.',
  Empirical_80_Coverage_pct:'Percentage of test days where actual demand fell inside the 80% interval band.',
  Empirical_95_Coverage_pct:'Percentage of test days where actual demand fell inside the 95% interval band.'
};
const TABLE_CATEGORIES={
  'All':'All Tables',
  'Leaderboard':'1. Leaderboard & Accuracy',
  'Forecasts':'2. Future & Weekly Forecasts',
  'Exploration':'3. Exploration & Data Prep',
  'Diagnostics':'4. Tuning & Diagnostics',
  'Inventory':'5. Inventory & EOQ'
};
function tableCategoryFor(key){
  if(['21_cv_heatmap','23_actual_vs_forecast'].includes(key))return 'Leaderboard';
  if(key==='22_demand_calendar_heatmap')return 'Exploration';
  if(key==='24_error_direction_matrix')return 'Diagnostics';
  if(['model_leaderboard','validation_metrics','model_metrics','cv_summary','cv_fold_metrics','diebold_mariano','interval_coverage','ensemble_weights','multi_origin_metrics'].includes(key))return 'Leaderboard';
  if(['weekly_forecast_summary','forecast','forecast_5_steps','test_predictions'].includes(key))return 'Forecasts';
  if(['input_data','raw_input_data','data_preparation_log','descriptive_statistics','stl_decomposition','stationarity_tests','distribution_summary','demand_acf_pacf','weekday_gaps','treatment_comparison'].includes(key))return 'Exploration';
  if(['sma_tuning','model_tuning','feature_importance','feature_comparison','sarima_aic_search','regression_coefficients','lead_time_errors'].includes(key))return 'Diagnostics';
  return 'Inventory';
}

const state={
  headers:[],raw:[],rows:[],name:'',result:null,version:0,runVersion:-1,
  busy:false,mode:'browser',worker:null,dataPage:0,tablePage:0,graphGroup:'All',
  hasRun:false,pendingRun:false,editTimer:null,
  fastModeUserTouched:false,
  selectedOverviewKeys:new Set(),
  selectedGalleryKeys:new Set(),
  galleryCols:1,
  tableCategory:'All',
  tableSortCol:null,
  tableSortAsc:true,
  tableSearch:'',
  dataFilter:'all',
  purchaseDecision:null,tableFull:false
};
const weekdayNames={2:'Monday',3:'Tuesday',4:'Wednesday',5:'Thursday',6:'Friday',7:'Saturday',8:'Sunday'};
function usesAllCalendarDays(){return $('calendar-mode')?.value==='all_days';}
function calendarCycleLength(){return usesAllCalendarDays()?7:5;}
function nextCalendarCode(day){const length=calendarCycleLength();return (Number(day)-1)%length+2;}
const fmt=(value,digits=2)=>value===null||value===undefined?'—':typeof value==='number'?value.toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits}):String(value);

function fmtCell(value,colName=''){
  if(value===null||value===undefined||value==='')return '—';
  if(typeof value==='boolean')return value?'Yes':'No';
  if(typeof value!=='number')return String(value);
  if(!Number.isFinite(value))return '—';
  if(INTEGER_COLUMNS.has(colName)||Number.isInteger(value)&&(/^(Observation|Rank|Fold|Lag|Step|Count|Bin|Window|Origin|Order_Number|Operating_)/i.test(colName))){
    return Math.round(value).toLocaleString('en-US');
  }
  if(FOUR_DEC_COLUMNS.has(colName))return fmt(value,4);
  if(THREE_DEC_COLUMNS.has(colName))return fmt(value,3);
  return fmt(value,2);
}

const normalized=s=>String(s).toLowerCase().replace(/[^a-z0-9]/g,'');
function el(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
function error(message){$('error').textContent=message;$('error').hidden=!message;}

function activeDataTypeLabel(){
  const dt=$('data-type').value;
  if(dt==='Custom'){
    const custom=$('custom-data-type').value.trim();
    return custom||'Custom series';
  }
  return dt;
}
function activeUnitLabel(){
  return $('unit-name').value.trim()||'units';
}
function displayLabelWithUnit(dtype=activeDataTypeLabel(),unit=activeUnitLabel()){
  if(!unit||dtype.trim().toLowerCase()===unit.trim().toLowerCase())return dtype.trim();
  return `${dtype.trim()} (${unit.trim()})`;
}

function syncDefaultFastMode(){
  if(!$('fast-mode')||state.fastModeUserTouched)return;
  const thr=Number($('fast-mode-threshold')?.value||500);
  $('fast-mode').checked=state.rows.length>thr;
}

function updateLargeFileWarning(){
  const warning=$('large-file-warning');
  if(!warning)return;
  if(state.rows.length>LARGE_FILE_WARNING_ROWS){
    warning.textContent=`Large file: ${state.rows.length.toLocaleString('en-US')} observations. Fast mode is recommended and turns on automatically above ${$('fast-mode-threshold')?.value||500} rows.`;
    warning.hidden=false;
  }else{
    warning.textContent='';
    warning.hidden=true;
  }
}

function updateProgressUI(message,percent=null){
  const wrap=$('progress-wrap'),bar=$('progress-bar'),stepEl=$('progress-step'),track=$('progress-track');
  if(!wrap||!bar||!stepEl)return;
  if(!message){
    wrap.hidden=true;
    bar.style.width='0%';
    stepEl.textContent='';
    return;
  }
  wrap.hidden=false;
  stepEl.textContent=message;
  let pct=percent;
  if(pct===null){
    const m=/Step\s+(\d+)\s*\/\s*(\d+)/i.exec(message);
    if(m){
      pct=Math.round((Number(m[1])/Number(m[2]))*100);
    }else if(/complete/i.test(message)){
      pct=100;
    }else if(/loading/i.test(message)){
      pct=12;
    }else{
      pct=35;
    }
  }
  pct=Math.max(5,Math.min(100,pct));
  bar.style.width=`${pct}%`;
  if(track)track.setAttribute('aria-valuenow',String(pct));
}

function updateStepper(step){
  for(let i=1;i<=5;i++){
    const b=$('step-badge-'+i);
    if(b){
      b.classList.toggle('active',i===step);
      b.classList.toggle('done',i<step);
    }
  }
}

function promptDataTypeStep(){
  const box=$('datatype-step-box');
  if(box){
    box.classList.add('highlight-step');
    setTimeout(()=>box.classList.remove('highlight-step'),1800);
  }
  const promptEl=$('datatype-prompt');
  if(promptEl){
    promptEl.textContent=`Data loaded (${state.rows.length.toLocaleString('en-US')} rows). Confirm your data type and unit below, then click Generate Part 1 Forecast Report.`;
  }
  if($('pp-stock-unit-hint')){
    $('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
  }
  updateStepper(2);
}

function clearResults(){
  state.result=null;
  $('result-overview').hidden=true;
  $('initial').hidden=false;
  if($('fast-mode-banner')){$('fast-mode-banner').textContent='';$('fast-mode-banner').hidden=true;}
  if($('model-warning')){$('model-warning').textContent='';$('model-warning').hidden=true;}
  if($('cv-warning')){$('cv-warning').textContent='';$('cv-warning').hidden=true;}
  if($('interval-horizon-note'))$('interval-horizon-note').textContent='';
  if($('executive-summary-card'))$('executive-summary-card').replaceChildren();
  if($('leaderboard-section'))$('leaderboard-section').replaceChildren();
  if($('exploration-prep-panels'))$('exploration-prep-panels').replaceChildren();
  if($('inline-comparison-tables'))$('inline-comparison-tables').replaceChildren();
  $('featured').replaceChildren();
  if($('featured-methods'))$('featured-methods').replaceChildren();
  if($('featured-interpretation'))$('featured-interpretation').replaceChildren();
  if($('overview-graph-picker'))$('overview-graph-picker').replaceChildren();
  if($('overview-gallery'))$('overview-gallery').replaceChildren();
  if($('dialog-methods'))$('dialog-methods').replaceChildren();
  if($('dialog-interpretation'))$('dialog-interpretation').replaceChildren();
  $('graph-filters').replaceChildren();
  if($('graph-checkboxes'))$('graph-checkboxes').replaceChildren();
  $('graph-gallery').replaceChildren(el('p','Run Part 1 first to generate charts from your data.','empty'));
  $('results-table').replaceChildren(el('p','Run Part 1 to calculate result tables.','empty'));
  $('table-pages').replaceChildren();
  $('table-select').replaceChildren();
  $('table-select').disabled=true;
  if($('table-row-search'))$('table-row-search').disabled=true;
  $('download-table').disabled=true;
  $('download-all').disabled=true;
  $('pp-results').hidden=true;
  const graphsTab=document.querySelector('[data-view="graphs"]');
  if(graphsTab)graphsTab.textContent='All Graphs';
}

function scheduleAutoRun(immediate=false){
  if(!state.hasRun||state.rows.length<40)return;
  if(state.editTimer){clearTimeout(state.editTimer);state.editTimer=null;}
  if(immediate){
    if(state.busy)state.pendingRun=true;
    else runAnalysis(false);
  }else{
    state.editTimer=setTimeout(()=>{
      state.editTimer=null;
      if(state.busy)state.pendingRun=true;
      else runAnalysis(false);
    },650);
  }
}

function dirty(autoImmediate=false,keepExistingView=true){
  state.version++;
  if(!keepExistingView&&state.result)clearResults();
  $('dirty').hidden=!state.hasRun;
  $('status').textContent=state.hasRun?'Inputs changed. Click "Update Report Now" or wait a moment…':'Step 2: Confirm your data type and unit, then click Generate Part 1 Forecast Report.';
  drawPreview();
  if(autoImmediate)scheduleAutoRun(true);
}

$('dirty-rerun-btn')?.addEventListener('click',()=>runAnalysis(false));

function view(id){
  document.querySelectorAll('.view').forEach(x=>x.hidden=x.id!==id);
  document.querySelectorAll('[data-view]').forEach(x=>{
    x.classList.toggle('active',x.dataset.view===id);
    x.setAttribute('aria-selected',String(x.dataset.view===id)); x.tabIndex=x.dataset.view===id?0:-1;
  });
  if(id==='data')renderInput();
  if(id==='purchase'){
    $('pp-form').hidden=false;
    if($('pp-stock-unit-hint'))$('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
  }
}

function setSettingsCollapsed(collapsed){
  const workspace=$('workspace'),toggle=$('settings-toggle'),topBtn=$('topbar-settings-btn');
  if(!workspace)return;
  workspace.classList.toggle('settings-collapsed',collapsed);
  if(toggle){
    toggle.textContent=collapsed?'Show settings':'Hide settings';
    toggle.setAttribute('aria-expanded',String(!collapsed));
  }
  if(topBtn){
    topBtn.textContent=collapsed?'⚙ Show settings':'⚙ Hide settings';
    topBtn.setAttribute('aria-expanded',String(!collapsed));
  }
}
$('settings-toggle')?.addEventListener('click',()=>setSettingsCollapsed(!$('workspace').classList.contains('settings-collapsed')));
$('topbar-settings-btn')?.addEventListener('click',()=>setSettingsCollapsed(!$('workspace').classList.contains('settings-collapsed')));
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>view(b.dataset.view)));

$('data-type').addEventListener('change',()=>{
  const isCustom=$('data-type').value==='Custom';
  $('custom-type-wrap').hidden=!isCustom;
  if(!isCustom&&DEFAULT_UNITS[$('data-type').value]){
    $('unit-name').value=DEFAULT_UNITS[$('data-type').value];
  }
  if($('pp-stock-unit-hint'))$('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
  dirty(false,true);
});
$('custom-data-type').addEventListener('input',()=>dirty(false,true));
$('unit-name').addEventListener('input',()=>{
  if($('pp-stock-unit-hint'))$('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
  dirty(false,true);
});
document.querySelectorAll('.unit-chips .chip').forEach(btn=>{
  btn.addEventListener('click',()=>{
    $('unit-name').value=btn.dataset.unit;
    if($('pp-stock-unit-hint'))$('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
    dirty(false,true);
  });
});

document.querySelectorAll('.service-chips .svc-chip').forEach(btn=>{
  btn.addEventListener('click',()=>{
    if($('pp-service'))$('pp-service').value=btn.dataset.svc||'';
    document.querySelectorAll('.service-chips .svc-chip').forEach(c=>c.classList.toggle('active-chip',c===btn));
  });
});

