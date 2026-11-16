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

function parseCSV(text){
  text=text.replace(/^\uFEFF/,'');
  const first=text.split(/\r?\n/,1)[0];
  const delimiter=[';',',','\t'].sort((a,b)=>first.split(b).length-first.split(a).length)[0];
  const records=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
    else if(c===delimiter&&!quoted){row.push(field);field='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x.trim()))records.push(row);row=[];field='';}
    else field+=c;
  }
  if(quoted)throw new Error('A quoted CSV value is not closed. Check the file format.');
  row.push(field);if(row.some(x=>x.trim()))records.push(row);
  if(records.length<2)throw new Error('The CSV needs a header row and at least 40 data rows.');
  const headers=records.shift().map(x=>x.trim());
  if(headers.some(x=>!x)||new Set(headers).size!==headers.length)throw new Error('Use a unique, nonblank header for every column.');
  if(records.some(r=>r.length!==headers.length))throw new Error('Some CSV rows have a different number of columns. Check separators and quoted values.');
  if(state.mode==='browser'&&records.length>BROWSER_MAX_OBSERVATIONS)throw new Error(`The hosted browser version accepts up to ${BROWSER_MAX_OBSERVATIONS.toLocaleString('en-US')} observations per run (found ${records.length.toLocaleString('en-US')} rows).`);
  if(records.length>MAX_OBSERVATIONS)throw new Error(`The local launcher accepts up to ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations per run (found ${records.length.toLocaleString('en-US')} rows).`);
  return {headers,rows:records};
}

function loadCSV(text,name,triggerRun=false){
  const parsed=parseCSV(text);state.headers=parsed.headers;state.raw=parsed.rows;state.name=name;state.dataPage=0;
  state.fastModeUserTouched=false;
  const demand=$('demand-column'),calendar=$('calendar-column');demand.replaceChildren();calendar.replaceChildren(new Option('No calendar column: generate weekdays',''));
  state.headers.forEach((h,i)=>{demand.add(new Option(h,String(i)));calendar.add(new Option(h,String(i)));});
  const d=state.headers.findIndex(h=>['totaldemand','totalorders','targettotalorders','demand','sales','quantity'].includes(normalized(h)));
  demand.value=String(d<0?0:d);
  const c=state.headers.findIndex(h=>normalized(h).includes('dayoftheweek')||['dayofweek','weekday','day','date'].includes(normalized(h)));
  calendar.value=c<0?'':String(c);$('calendar-type').value=c>=0&&normalized(state.headers[c]).includes('date')?'date':'weekday';
  if(c>=0&&$('calendar-type').value==='date'){
    const hasWeekend=state.raw.some(row=>{const dt=new Date(String(row[c]).trim()+'T12:00:00Z');return !Number.isNaN(dt.getTime())&&[0,6].includes(dt.getUTCDay());});
    $('calendar-mode').value=hasWeekend?'all_days':'weekdays';
  }
  remap(triggerRun);error('');$('run').disabled=state.rows.length===0;
  promptDataTypeStep();
}

function remap(triggerRun=false){
  const d=Number($('demand-column').value),c=$('calendar-column').value;
  const type=$('calendar-type').value;
  state.rows=state.raw.map(row=>({demand:row[d],weekday:c!==''&&type==='weekday'?row[Number(c)]:'',date:c!==''&&type==='date'?row[Number(c)]:''}));
  $('calendar-type').disabled=c==='';$('start-weekday').disabled=c!=='';
  syncDefaultFastMode();
  updateLargeFileWarning();
  $('file-info').textContent=`${state.name} · ${state.rows.length.toLocaleString('en-US')} observations`;
  state.dataPage=0;renderInput();dirty(Boolean(triggerRun),!triggerRun);
}

for(const id of ['demand-column','calendar-column','calendar-type'])$(id).addEventListener('change',()=>remap(true));
$('calendar-mode')?.addEventListener('change',()=>{renderInput();dirty(false,true);});
for(const id of ['horizon','cv-folds','min-interval-errors','missing-treatment','max-missing-pct','outlier-treatment','outlier-threshold','hw-seasonal','hw-damped']){
  if($(id))$(id).addEventListener('change',()=>dirty(false,true));
}
if($('fast-mode')){
  $('fast-mode').addEventListener('change',()=>{
    state.fastModeUserTouched=true;
    dirty(false,true);
  });
}
if($('fast-mode-threshold')){
  $('fast-mode-threshold').addEventListener('change',()=>{
    state.fastModeUserTouched=false;
    syncDefaultFastMode();
    dirty(false,true);
  });
}
$('start-weekday').addEventListener('change',()=>{renderInput();dirty(false,true);});

async function sample(isUserAction=false){
  try{
    $('status').textContent='Loading project dataset…';
    const r=await fetch('sample.csv');
    if(!r.ok)throw new Error('The project sample could not load. Upload a CSV instead.');
    loadCSV(await r.text(),'UCI project dataset',isUserAction);
  }catch(e){error(e.message);}
}
$('sample').addEventListener('click',()=>sample(true));
$('file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;await receiveFile(f);e.target.value='';});

async function receiveFile(f){
  if(state.busy)return;
  error('');
  if(f.size>MAX_FILE_BYTES){
    const sizeMB=(f.size/(1024*1024)).toFixed(1);
    const limitMB=(MAX_FILE_BYTES/(1024*1024)).toFixed(0);
    const msg=`File "${f.name}" is ${sizeMB} MB, which exceeds the ${limitMB} MB maximum upload limit. Choose a CSV or TSV up to ${limitMB} MB (and up to ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations).`;
    error(msg);
    $('status').textContent='Upload rejected: file exceeds the maximum size limit.';
    return;
  }
  try{
    const sizeLabel=f.size>=1_000_000?`${(f.size/(1024*1024)).toFixed(2)} MB`:`${Math.max(1,Math.round(f.size/1024))} KB`;
    $('status').textContent=`Reading ${f.name} (${sizeLabel})…`;
    $('file-info').textContent=`Loading ${f.name} (${sizeLabel})…`;
    await new Promise(resolve=>setTimeout(resolve,0));
    const text=await f.text();
    $('status').textContent=`Parsing ${f.name} (${sizeLabel})…`;
    await new Promise(resolve=>setTimeout(resolve,0));
    loadCSV(text,f.name,true);
  }catch(e){
    error(e.message);
    $('status').textContent='Could not load the uploaded file. Check the error message above.';
  }
}

$('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('dragging'));
$('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('dragging');if(e.dataTransfer.files[0])receiveFile(e.dataTransfer.files[0]);});

function drawPreview(){
  const holder=$('preview-plot');if(!holder)return;
  holder.replaceChildren(el('p',state.rows.length?'Your data is loaded. Open setup, confirm its type and unit, then generate the forecast report.':'Load a CSV/TSV or the project sample to begin.'));
  $('preview-heading').textContent=state.rows.length?'Dataset ready for analysis':'Start with your daily demand data';
}

function paginate(container,page,count,onChange){
  container.replaceChildren();
  if(count<=25)return;
  const total=Math.ceil(count/25);
  const prev=el('button','Previous'),next=el('button','Next');
  prev.disabled=page===0;next.disabled=page===total-1;
  prev.onclick=()=>onChange(page-1);next.onclick=()=>onChange(page+1);
  container.append(prev,el('span',`Page ${page+1} of ${total} · ${count.toLocaleString('en-US')} rows`),next);
}

function syncRowToRaw(rowIndex){
  if(!state.raw[rowIndex])return;
  const d=Number($('demand-column').value),c=$('calendar-column').value;
  const type=$('calendar-type').value;
  state.raw[rowIndex][d]=state.rows[rowIndex].demand;
  if(c!==''){
    state.raw[rowIndex][Number(c)]=type==='date'?state.rows[rowIndex].date:state.rows[rowIndex].weekday;
  }
}

function renderDataHealthBar(){
  const holder=$('data-health-bar');if(!holder)return;holder.replaceChildren();
  const x=el('div',undefined,'health-item');x.append(el('span','Loaded observations'),el('strong',`${state.rows.length.toLocaleString('en-US')} rows`));
  holder.append(x,el('p','Edit raw observations below. Recalculate to refresh the engine’s statistics and forecasts.'));
}

function renderInput(){
  const container=$('input-table');if(!container)return;
  container.replaceChildren();
  renderDataHealthBar();
  const dtype=activeDataTypeLabel();const unit=activeUnitLabel();
  const disp=displayLabelWithUnit(dtype,unit);
  const table=el('table'),thead=el('thead'),head=el('tr');
  ['Observation',disp,'Calendar','Action'].forEach(x=>head.append(el('th',x)));
  thead.append(head);table.append(thead);
  const body=el('tbody');
  const automatic=$('calendar-column').value==='';
  const isDate=$('calendar-type').value==='date'&&!automatic;

  const indexedRows=state.rows.map((r,idx)=>({r,idx})).filter(({r})=>{
    if(state.dataFilter==='blanks')return r.demand===''||r.demand===null||Number.isNaN(Number(r.demand));
    return true;
  });

  const maxPage=Math.max(0,Math.ceil(indexedRows.length/25)-1);
  if(state.dataPage>maxPage)state.dataPage=maxPage;

  indexedRows.slice(state.dataPage*25,state.dataPage*25+25).forEach(({r:row,idx:i})=>{
    const tr=el('tr');
    tr.append(el('td',String(i+1)));
    const cell=el('td'),input=el('input');
    input.type='number';input.min='0';input.step='any';input.value=row.demand;
    input.setAttribute('aria-label',`${dtype} at observation ${i+1}`);
    input.disabled=state.busy;
    input.oninput=()=>{
      row.demand=input.value;
      syncRowToRaw(i);
      if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      renderDataHealthBar();
      drawPreview();
    };
    input.onchange=()=>dirty(false,true);
    cell.append(input);tr.append(cell);

    const day=el('td');
    if(automatic){
      day.textContent=weekdayNames[(Number($('start-weekday').value)-2+i)%calendarCycleLength()+2]+' (generated)';
    }else{
      const cInput=el('input');
      cInput.type=isDate?'date':'text';
      cInput.value=isDate?row.date:row.weekday;
      cInput.setAttribute('aria-label',`Calendar at observation ${i+1}`);
      cInput.disabled=state.busy;
      cInput.oninput=()=>{
        row[isDate?'date':'weekday']=cInput.value;
        syncRowToRaw(i);
        if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      };
      cInput.onchange=()=>dirty(false,true);
      day.append(cInput);
    }
    tr.append(day);

    const last=el('td'),remove=el('button','Remove','remove-row');
    remove.disabled=state.busy;
    remove.setAttribute('aria-label',`Remove observation ${i+1}`);
    remove.onclick=()=>{
      state.rows.splice(i,1);
      if(state.raw.length>i)state.raw.splice(i,1);
      state.dataPage=Math.min(state.dataPage,Math.max(0,Math.ceil(state.rows.length/25)-1));
      syncDefaultFastMode();renderInput();promptDataTypeStep();
      if($('apply-data-edits'))$('apply-data-edits').hidden=false;
      dirty(false,true);
    };
    last.append(remove);tr.append(last);body.append(tr);
  });
  table.append(body);
  container.append(table);
  if(!indexedRows.length){
    container.append(el('p','No rows match this filter.','empty'));
  }
  paginate($('data-pages'),state.dataPage,indexedRows.length,p=>{state.dataPage=p;renderInput();});
  const calendarText=usesAllCalendarDays()?'Monday–Sunday calendar days':'Monday–Friday operating days';
  $('data-description').textContent=`${state.rows.length.toLocaleString('en-US')} observations of ${disp.toLowerCase()}. ${automatic?`Consecutive ${calendarText} will be generated.`:isDate?`Dates must be consecutive ${calendarText}.`:usesAllCalendarDays()?'Weekdays: Monday–Sunday names or codes 2–8.':'Weekdays: Monday–Friday names or UCI codes 2–6.'}`;
  $('file-info').textContent=`${state.name} · ${state.rows.length.toLocaleString('en-US')} observations`;
}

document.querySelectorAll('#data-row-filters button').forEach(btn=>{
  btn.addEventListener('click',()=>{
    state.dataFilter=btn.dataset.dfilter||'all';
    state.dataPage=0;
    document.querySelectorAll('#data-row-filters button').forEach(b=>b.classList.toggle('active',b===btn));
    renderInput();
  });
});

$('data-jump-input')?.addEventListener('change',()=>{
  const target=Number($('data-jump-input').value);
  if(Number.isFinite(target)&&target>=1&&target<=state.rows.length){
    state.dataFilter='all';
    document.querySelectorAll('#data-row-filters button').forEach(b=>b.classList.toggle('active',b.dataset.dfilter==='all'));
    state.dataPage=Math.floor((target-1)/25);
    renderInput();
  }
});

$('apply-data-edits')?.addEventListener('click',()=>{
  $('apply-data-edits').hidden=true;
  runAnalysis(false);
});

$('add-row').onclick=()=>{
  if(state.rows.length>=MAX_OBSERVATIONS)return error(`Maximum ${MAX_OBSERVATIONS.toLocaleString('en-US')} observations.`);
  let row={demand:state.rows.at(-1)?.demand||'0',weekday:'',date:''};
  const last=state.rows.at(-1);
  if($('calendar-column').value!==''){
    if($('calendar-type').value==='date'&&last?.date){
      const d=new Date(last.date+'T12:00:00Z');
      do{d.setUTCDate(d.getUTCDate()+1);}while(!usesAllCalendarDays()&&[0,6].includes(d.getUTCDay()));
      row.date=d.toISOString().slice(0,10);
    }else{
      const raw=String(last?.weekday||'6').toLowerCase();
      const current=Number(raw)||Object.entries(weekdayNames).find(([,v])=>v.toLowerCase().slice(0,3)===raw.slice(0,3))?.[0]||6;
      row.weekday=String(nextCalendarCode(current));
    }
  }
  state.rows.push(row);
  state.dataPage=Math.floor((state.rows.length-1)/25);
  syncDefaultFastMode();renderInput();promptDataTypeStep();
  if($('apply-data-edits'))$('apply-data-edits').hidden=false;
  dirty(false,true);
};

function csvCell(value){return '"'+String(value??'').replaceAll('"','""')+'"';}
function inputCSV(){const automatic=$('calendar-column').value==='';const dateMode=$('calendar-type').value==='date'&&!automatic;const name=dateMode?'Date':'Weekday';return [`${activeDataTypeLabel()},${name}`,...state.rows.map((r,i)=>[r.demand,automatic?weekdayNames[(Number($('start-weekday').value)-2+i)%calendarCycleLength()+2]:dateMode?r.date:r.weekday].map(csvCell).join(','))].join('\n');}
function download(blob,name){const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function fromBase64(base64,type){const binary=atob(base64);return new Blob([Uint8Array.from(binary,c=>c.charCodeAt(0))],{type});}
$('download-input').onclick=()=>download(new Blob([inputCSV()],{type:'text/csv;charset=utf-8'}),'edited_demand.csv');

function activeFigureView(figure){
  if(figure.method_views){
    const key=figure.activeMethod||figure.default_method||'SMA';
    if(figure.method_views[key])return figure.method_views[key];
  }
  return {
    title:figure.title,
    data:figure.data,
    method:null,
    accent_color:figure.accent_color||'#2563EB',
    interpretation:figure.interpretation||{shows:'',how_to_read:'',understood:'',next_step:''}
  };
}

function renderInterpretationBox(container,interp,accentColor){
  container.replaceChildren(); if(!interp)return;
  for(const [i,key,label] of [[1,'shows','What this graph is'],[2,'how_to_read','How to see and read it'],[3,'understood','What we understood'],[4,'next_step','What to do next']]){
    const d=el('details',undefined,'guide-part');d.open=i===3;
    d.append(el('summary',`${i}. ${label}`),el('p',interp[key]||'No additional interpretation is available.'));container.append(d);
  }
}

function buildMethodButtons(figure,onSelect){
  const bar=el('div',undefined,'method-buttons');
  bar.setAttribute('role','group');
  bar.setAttribute('aria-label',`Select forecasting method for ${figure.title}`);
  if(!figure.method_views)return bar;
  const current=figure.activeMethod||figure.default_method||'SMA';
  for(const [key,label] of METHOD_BUTTONS){
    if(!figure.method_views[key])continue;
    const btn=el('button',label,key===current?'active':'');
    btn.type='button';
    btn.dataset.method=key;
    const vColor=figure.method_views[key].accent_color||figure.accent_color||'#2563EB';
    if(key===current){
      btn.style.backgroundColor=vColor;
      btn.style.borderColor=vColor;
      const rgb=vColor.replace('#','').match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);const lum=rgb.reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);btn.style.color=(lum+.05)/.05>=1.05/(lum+.05)?'#000':'#fff';
    }
    btn.setAttribute('aria-pressed',String(key===current));
    btn.onclick=()=>{
      figure.activeMethod=key;
      onSelect(key);
    };
    bar.append(btn);
  }
  return bar;
}

function graphDownload(figure){
  const v=activeFigureView(figure);
  const baseName=figure.path.split('/').at(-1).replace(/\.png$/i,'');
  const suffix=v.method?`_${v.method}.png`:'.png';
  download(fromBase64(v.data.split(',')[1],'image/png'),baseName+suffix);
}

let dialogFigure;
function syncDialog(){
  if(!dialogFigure)return;
  const v=activeFigureView(dialogFigure);
  $('dialog-title').textContent=v.title;
  $('dialog-image').src=v.data;
  $('dialog-image').alt=chartAlt(dialogFigure,v); resetChartZoom(); $('open-full-size').href=URL.createObjectURL(fromBase64(v.data.split(',')[1],'image/png'));
  const holder=$('dialog-methods');
  if(holder){
    holder.replaceChildren();
    if(dialogFigure.method_views){
      const bar=buildMethodButtons(dialogFigure,()=>{
        syncDialog();
        if(dialogFigure.key==='02_evaluation')syncFeatured();
        renderOverviewCards();
        renderGallery();
      });
      holder.append(...bar.childNodes);
    }
  }
  if($('dialog-interpretation')){
    renderInterpretationBox($('dialog-interpretation'),v.interpretation||dialogFigure.interpretation,v.accent_color||dialogFigure.accent_color);
  }
}

function openGraph(figure){
  dialogFigure=figure;
  syncDialog();
  $('graph-dialog').showModal();
}
$('dialog-close').onclick=()=>$('graph-dialog').close();
$('dialog-download').onclick=()=>graphDownload(dialogFigure);
$('graph-dialog').addEventListener('click',e=>{if(e.target===$('graph-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});

function syncFeatured(){
  if(!state.result)return;
  const f=state.result.figures.find(x=>x.key==='02_evaluation');
  if(!f)return;
  const v=activeFigureView(f);
  const accent=v.accent_color||f.accent_color||'#2563EB';
  if($('featured-card'))$('featured-card').style.borderTopColor=accent;
  const image=el('img');image.src=v.data;image.alt=chartAlt(f,v);
  $('featured').replaceChildren(image);
  $('featured').onclick=()=>openGraph(f);
  if($('enlarge-featured'))$('enlarge-featured').onclick=()=>openGraph(f);
  $('download-featured').onclick=()=>graphDownload(f);
  if($('featured-title'))$('featured-title').textContent=v.title;
  const holder=$('featured-methods');
  if(holder){
    holder.replaceChildren();
    if(f.method_views){
      const bar=buildMethodButtons(f,()=>{
        syncFeatured();
        renderOverviewCards();
        renderGallery();
      });
      holder.append(...bar.childNodes);
    }
  }
  if($('featured-interpretation')){
    renderInterpretationBox($('featured-interpretation'),v.interpretation||f.interpretation,accent);
  }
}

function buildGraphCard(f,onMethodChange){
  const v=activeFigureView(f);
  const accent=v.accent_color||f.accent_color||'#2563EB';
  const article=el('article',undefined,'graph-tile graph-card');
  article.style.borderTop=`4px solid ${accent}`;

  const header=el('div',undefined,'graph-card-header');
  const badge=el('span',f.group,'graph-badge');
  badge.style.backgroundColor=accent+'18';
  badge.style.color=accent;
  const titleEl=el('h3',v.title);
  const leftHead=el('div',undefined,'graph-head-left');
  leftHead.append(badge,titleEl);

  const actions=el('div',undefined,'card-actions');
  const zoomBtn=el('button','Enlarge ⤢','secondary small-btn');
  zoomBtn.type='button';
  zoomBtn.onclick=()=>openGraph(f);
  const saveBtn=el('button','Download PNG','text-button');
  saveBtn.type='button';
  saveBtn.onclick=()=>graphDownload(f);
  actions.append(zoomBtn,saveBtn);
  header.append(leftHead,actions);

  const button=el('button',undefined,'graph-open');
  button.type='button';
  const img=el('img');
  img.src=v.data;img.alt=chartAlt(f,v);img.loading='lazy';
  button.setAttribute('aria-label','Enlarge '+v.title);
  button.onclick=()=>openGraph(f);
  button.append(img);

  const interpDetails=el('details',undefined,'graph-explanation');
  interpDetails.open=true;
  interpDetails.append(el('summary','What this graph means — How to see, read & understand it'));
  const interpBox=el('div',undefined,'interpretation-box');
  renderInterpretationBox(interpBox,v.interpretation||f.interpretation,accent);
  interpDetails.append(interpBox);

  article.append(header,button);
  if(f.method_views){
    const bar=buildMethodButtons(f,()=>{
      if(f.key==='02_evaluation')syncFeatured();
      onMethodChange();
    });
    article.append(bar);
  }
  article.append(interpDetails);
  return article;
}

function renderGraphPicker(container,figures,selectedSet,onChange){
  if(!container)return;
  container.replaceChildren();
  figures.forEach(f=>{
    const accent=f.accent_color||'#2563EB';
    const label=el('label',undefined,'picker-chip'+(selectedSet.has(f.key)?' active':''));
    if(selectedSet.has(f.key)){
      label.style.borderColor=accent;
      label.style.backgroundColor=accent+'12';
    }
    const cb=el('input');
    cb.type='checkbox';
    cb.checked=selectedSet.has(f.key);
    cb.onchange=()=>{
      if(cb.checked)selectedSet.add(f.key);
      else selectedSet.delete(f.key);
      onChange();
    };
    const dot=el('span',undefined,'picker-dot');
    dot.style.backgroundColor=accent;
    const text=el('span',f.title.split('(')[0].trim());
    label.append(cb,dot,text);
    container.append(label);
  });
}

function renderOverviewCards(){
  const container=$('overview-gallery');
  if(!container||!state.result)return;
  container.replaceChildren();
  const part1Figs=state.result.figures.filter(f=>f.part!==2);
  renderGraphPicker($('overview-graph-picker'),part1Figs,state.selectedOverviewKeys,()=>renderOverviewCards());
  const visible=part1Figs.filter(f=>state.selectedOverviewKeys.has(f.key));
  if(!visible.length){
    container.append(el('p','No graphs selected. Check any graph above or click "Select all graphs" to view report cards.','empty'));
    return;
  }
  visible.forEach(f=>{
    container.append(buildGraphCard(f,()=>{
      renderOverviewCards();
      renderGallery();
    }));
  });
}

if($('select-all-overview')){
  $('select-all-overview').onclick=()=>{
    if(!state.result)return;
    const part1Figs=state.result.figures.filter(f=>f.part!==2);
    const allSelected=part1Figs.every(f=>state.selectedOverviewKeys.has(f.key));
    if(allSelected)state.selectedOverviewKeys.clear();
    else part1Figs.forEach(f=>state.selectedOverviewKeys.add(f.key));
    $('select-all-overview').textContent=allSelected?'Select all graphs':'Clear selection';
    renderOverviewCards();
  };
}

function getGalleryCategoryPool(){
  if(!state.result)return [];
  return state.result.figures.filter(f=>{
    if(f.part===2&&!state.result.purchase_plan)return false;
    return state.graphGroup==='All'||f.group===state.graphGroup;
  });
}

function renderGallery(){
  const container=$('graph-gallery');
  if(!container)return;
  container.replaceChildren();
  if(!state.result)return;
  container.classList.toggle('two-col-cards',state.galleryCols===2);
  container.classList.toggle('single-col-cards',state.galleryCols!==2);

  const query=($('graph-search')?.value||'').trim().toLowerCase();
  const categoryPool=getGalleryCategoryPool();
  const pool=categoryPool.filter(f=>!query||`${f.title} ${f.group} ${f.key}`.toLowerCase().includes(query));
  const figures=pool.filter(f=>state.selectedGalleryKeys.has(f.key)&&(state.galleryAll||query||state.graphGroup!=='All'||KEY_CHARTS.has(f.key)));
  if($('show-all-charts')){$('show-all-charts').hidden=Boolean(state.guided);$('show-all-charts').textContent=state.galleryAll?'Show key charts':'Show all charts';$('show-all-charts').setAttribute('aria-pressed',String(Boolean(state.galleryAll)));}
  const searchMsg=$('graph-search-message');
  if(searchMsg){
    searchMsg.textContent=query&&!pool.length
      ?`That graph is not available. Supported choices in this category: ${categoryPool.map(f=>f.title.split('(')[0].trim()).join('; ') || 'none'}.`
      :query?`${pool.length} matching charts; ${figures.length} selected.`:`Showing ${figures.length} of ${categoryPool.length} charts in ${state.graphGroup}.`;
  }
  renderGraphPicker($('graph-checkboxes'),pool,state.selectedGalleryKeys,()=>renderGallery());
  const allSel=pool.length>0&&pool.every(f=>state.selectedGalleryKeys.has(f.key));
  if($('select-all-graphs'))$('select-all-graphs').textContent=allSel?'Clear selection':'Select all';
  if(!figures.length){
    container.append(el('p',query&&!pool.length?'No charts match this search. Clear the search or choose another category.':'No graphs selected in this category. Click "Select all" or check a graph above.','empty'));
    return;
  }
  figures.forEach(f=>{
    container.append(buildGraphCard(f,()=>{
      renderGallery();
      renderOverviewCards();
    }));
  });
}

$('graph-search')?.addEventListener('input',renderGallery);

$('gallery-cols-1')?.addEventListener('click',()=>{
  state.galleryCols=1;
  $('gallery-cols-1').classList.add('active-view-btn');
  $('gallery-cols-2')?.classList.remove('active-view-btn');
  renderGallery();
});
$('gallery-cols-2')?.addEventListener('click',()=>{
  state.galleryCols=2;
  $('gallery-cols-2').classList.add('active-view-btn');
  $('gallery-cols-1')?.classList.remove('active-view-btn');
  renderGallery();
});

if($('select-all-graphs')){
  $('select-all-graphs').onclick=()=>{
    if(!state.result)return;
    const pool=getGalleryCategoryPool();
    const allSelected=pool.every(f=>state.selectedGalleryKeys.has(f.key));
    if(allSelected)pool.forEach(f=>state.selectedGalleryKeys.delete(f.key));
    else pool.forEach(f=>state.selectedGalleryKeys.add(f.key));
    $('select-all-graphs').textContent=allSelected?'Select all':'Clear selection';
    renderGallery();
  };
}

function renderExecutiveSummary(result){
  const holder=$('executive-summary-card');
  if(!holder)return;
  holder.replaceChildren();
  const s=result.summary;
  const ex=s.executive_summary||{};
  const unit=s.unit_name||activeUnitLabel();
  const dayLabel=s.parameters?.calendar_mode==='all_days'?'Calendar Days':'Operating Days';
  const wape=Number(s.test_metrics?.WAPE_pct??ex.test_wape_pct??0);
  const mase=Number(s.test_metrics?.MASE??ex.test_mase??1);
  const accLabel=wape<10?'High accuracy (WAPE < 10%)':(wape<20?'Good operational accuracy':'Moderate / noisy series');

  const head=el('div',undefined,'exec-summary-header');
  head.append(
    el('h3',`Quick Takeaway Summary — ${s.selected_label} Selected as Best Model`),
    el('span',`${accLabel} · ${mase<1?'Beats naive benchmark':'Close to naive benchmark'}`,'metric-pill '+(mase<1?'good':'neutral'))
  );

  const grid=el('div',undefined,'exec-grid');
  const items=[
    ['1. Winning Model & Error',`${s.selected_label}`,`Test MAE: ${fmt(s.test_metrics.MAE)} ${unit} (WAPE: ${fmt(wape,1)}%, MASE: ${fmt(mase,3)})`],
    [`2. Next 5 ${dayLabel}`,`${fmt(ex.next_5_days_total)} ${unit} total`,`Daily average: ${fmt(ex.next_5_days_mean)} ${unit}/day over next week`],
    [`3. Full ${ex.horizon_days||60}-Day Horizon`,`${fmt(ex.horizon_total)} ${unit} total`,`Daily average: ${fmt(ex.horizon_mean)} ${unit}/day across ${ex.horizon_days||60} days`],
    ['4. Weekday Peak & EOQ Batch',`Peak: ${ex.peak_weekday||'—'} (${fmt(ex.peak_weekday_mean)} ${unit})`,`Lowest: ${ex.low_weekday||'—'} (${fmt(ex.low_weekday_mean)} ${unit}) · Base EOQ: ${fmt(s.final_inventory?.EOQ)} ${unit}`]
  ];
  for(const [kicker,big,sub] of items){
    const card=el('div',undefined,'exec-item');
    card.append(el('span',kicker),el('strong',big),el('small',sub));
    grid.append(card);
  }
  holder.append(head,grid);
}

function renderLeaderboard(result){
  const holder=$('leaderboard-section');
  if(!holder)return;
  holder.replaceChildren();
  const s=result.summary;
  const unit=s.unit_name||activeUnitLabel();
  const rows=s.leaderboard||[];
  if(!rows.length)return;

  holder.append(
    el('div','FORECASTING TEAM LEADERBOARD','section-label'),
    el('h3',`All ${rows.length} Evaluated Forecasting Models (Ranked by Validation MAE)`),
    el('p',`The winning model (${s.selected_label}) and Top-3 Ensemble members are chosen strictly on the validation split before touching the final test set.`)
  );

  const scroll=el('div',undefined,'table-scroll');
  const tbl=el('table'),thead=el('thead'),trH=el('tr');
  ['Val Rank','Model','Role','Validation MAE',`Final Test MAE (${unit})`,'Test WAPE (%)','Test MASE','Test Bias','CV Mean MAE'].forEach(h=>trH.append(el('th',h)));
  thead.append(trH);tbl.append(thead);
  const tbody=el('tbody');
  rows.forEach(r=>{
    const tr=el('tr');
    if(r.Model===s.selected_model)tr.className='winner-row';
    else if(r.Role==='Top-3 Ensemble Member')tr.className='ensemble-row';

    const roleTd=el('td');
    const badge=el('span',r.Role=== 'Selected Winner'?'★ Selected Winner':r.Role,'role-badge '+(r.Role==='Selected Winner'?'winner':(r.Role.includes('Ensemble')?'member':'')));
    roleTd.append(badge);

    tr.append(
      el('td',`#${r.Validation_Rank}`),
      el('td',r.Model_Label),
      roleTd,
      el('td',fmt(r.Validation_MAE,2)),
      el('td',fmt(r.Test_MAE,2)),
      el('td',`${fmt(r.Test_WAPE_pct,2)}%`),
      el('td',fmt(r.Test_MASE,3)),
      el('td',fmt(r.Test_Bias,2)),
      el('td',r.CV_Mean_MAE!==null?fmt(r.CV_Mean_MAE,2):'—')
    );
    tbody.append(tr);
  });
  tbl.append(tbody);
  scroll.append(tbl);
  holder.append(scroll);
}

function buildMiniTableElement(tableObj,titleText,subtitleText){
  const wrap=el('div',undefined,'inline-mini-table');
  const h=el('h3',titleText);
  const p=el('p',subtitleText,'hint');
  const scroll=el('div',undefined,'table-scroll');
  const tbl=el('table'),thead=el('thead'),trH=el('tr');
  tableObj.columns.forEach(c=>trH.append(el('th',c.replaceAll('_',' '))));
  thead.append(trH);tbl.append(thead);
  const tbody=el('tbody');
  tableObj.rows.forEach(r=>{
    const tr=el('tr');
    tableObj.columns.forEach(c=>tr.append(el('td',fmtCell(r[c],c))));
    tbody.append(tr);
  });
  tbl.append(tbody);
  scroll.append(tbl);
  wrap.append(h,p,scroll);
  return wrap;
}

function renderSummaryPanelsAndInlineTables(result){
  const panels=$('exploration-prep-panels');
  const inlineTables=$('inline-comparison-tables');
  if(panels)panels.replaceChildren();
  if(inlineTables)inlineTables.replaceChildren();
  const s=result.summary;
  const unit=s.unit_name||activeUnitLabel();

  if(panels){
    const p1=el('div',undefined,'info-panel');
    p1.style.borderLeftColor='#0E7490';
    p1.append(
      el('div','DATA EXPLORATION','section-label'),
      el('h3','Stationarity, Distribution & ACF/PACF'),
      el('p',s.exploration?.stationarity_summary||''),
      el('p',`Distribution: mean ${fmt(s.exploration?.distribution?.mean)} ${unit}, median ${fmt(s.exploration?.distribution?.median)} ${unit}, skewness ${fmt(s.exploration?.distribution?.skewness,3)}.`)
    );
    panels.append(p1);

    const dp=s.data_preparation||{};
    const p2=el('div',undefined,'info-panel');
    p2.style.borderLeftColor='#D97706';
    const blankMsg=dp.blank_count>0
      ?`Blanks imputed: ${dp.blank_count} (${fmt(dp.blank_share_pct,1)}% of rows, positions: ${dp.blank_positions.join(', ')}) via ${dp.missing_treatment}.`
      :`Blank demand values: 0 (rule-of-thumb max allowed: ${dp.max_missing_pct_rule}%).`;
    const outMsg=dp.outlier_count>0
      ?`Causal one-step residual outliers (|robust z| > ${dp.outlier_threshold_rule} rule of thumb): ${dp.outlier_count} flagged at observation(s) ${dp.outlier_positions.slice(0,12).join(', ')} · action: ${dp.outlier_treatment}.`
      :`Causal one-step residual outliers (|robust z| > ${dp.outlier_threshold_rule} rule of thumb): 0 flagged.`;
    p2.append(
      el('div','DATA PREPARATION','section-label'),
      el('h3','Blank Values & Causal Outlier Treatment'),
      el('p',blankMsg),
      el('p',outMsg)
    );
    panels.append(p2);

    const tu=s.tuning||{},fi=s.feature_importance||{};
    const p3=el('div',undefined,'info-panel');
    p3.style.borderLeftColor='#7E22CE';
    const rfParamStr=tu.best_rf_params?`n_estimators=${tu.best_rf_params.n_estimators}, max_depth=${tu.best_rf_params.max_depth}, min_samples_leaf=${tu.best_rf_params.min_samples_leaf}`:'default';
    p3.append(
      el('div','FEATURES, TUNING & IMPORTANCE','section-label'),
      el('h3',`Richer Features (${tu.feature_names?.length||12} predictors)`),
      el('p',`Chosen on training/validation only: Ridge λ = ${fmt(tu.best_ridge_lambda,4)}; Random Forest (${rfParamStr}).`),
      el('p',`Top Ridge standardized feature: ${fi.top_ridge_feature||'—'} (${fmt(fi.top_ridge_coef,2)} ${unit}/SD).`+(fi.top_rf_feature?` Top RF validation permutation feature: ${fi.top_rf_feature} (ΔMAE ${fmt(fi.top_rf_perm_mae,2)} ${unit}).`:'')),
      el('p',fi.warning||'')
    );
    panels.append(p3);

    const p4=el('div',undefined,'info-panel');
    p4.style.borderLeftColor='#0F766E';
    const hw=s.holt_winters||{},sar=s.sarima||{};
    const sarText=sar.enabled
      ?`SARIMA${JSON.stringify(sar.order)}x${JSON.stringify(sar.seasonal_order)} selected by training AIC (${fmt(sar.best_aic,2)}); ${sar.skipped_count} of ${sar.candidates_tried} candidates skipped for non-convergence.`
      :(sar.skipped_by_fast_mode?'SARIMA skipped by Fast mode.':'SARIMA unavailable (statsmodels not loaded).');
    const hwText=hw.enabled
      ?`Holt-Winters (period=${hw.seasonal_periods}): seasonal=${hw.seasonal}, damped=${hw.damped_trend}.`
      :'Holt-Winters unavailable (statsmodels not loaded).';
    p4.append(
      el('div','NEW MODELS & ENSEMBLE','section-label'),
      el('h3',`Top-3 Validation Ensemble: ${(s.ensemble_member_labels||[]).join(', ')}`),
      el('p',hwText),
      el('p',sarText)
    );
    panels.append(p4);
  }

  if(inlineTables){
    const featTbl=result.tables.find(t=>t.key==='feature_comparison');
    const treatTbl=result.tables.find(t=>t.key==='treatment_comparison');
    if(featTbl||treatTbl){
      const drawer=el('details',undefined,'comparison-drawer');
      drawer.append(el('summary','View Baseline vs. Richer Features & Raw vs. Treated Accuracy Comparison Tables'));
      if(featTbl){
        drawer.append(buildMiniTableElement(
          featTbl,
          `Baseline vs. Richer Features Comparison (${unit})`,
          s.tuning?.feature_comparison_summary||'Comparison of old (baseline) and new (richer) feature sets on validation and final test.'
        ));
      }
      if(treatTbl){
        drawer.append(buildMiniTableElement(
          treatTbl,
          `Raw vs. Treated Data Comparison for Selected Model (${s.selected_label}, ${unit})`,
          'Compares validation and final test metrics when a blank-value or causal-outlier treatment was applied.'
        ));
      }
      inlineTables.append(drawer);
    }
  }
}

function renderPurchasePlanSection(){
  if(!state.result||!state.result.purchase_plan){
    $('pp-results').hidden=true;
    return;
  }
  const pp=state.result.purchase_plan;
  const unit=pp.unit_name||activeUnitLabel();
  $('pp-results').hidden=false;
  $('pp-disclaimer').textContent=pp.disclaimer;
  $('pp-eoq').textContent=`${fmt(pp.EOQ)} ${unit}`;
  if(pp.service_level_provided){
    $('pp-safety').textContent=`${fmt(pp.safety_stock)} ${unit}`;
    if($('pp-safety-note'))$('pp-safety-note').textContent=`Buffer for ${fmt(pp.service_level_pct,1)}% nominal service target`;
    if($('pp-reorder-note'))$('pp-reorder-note').textContent=`Lead-time demand + ${fmt(pp.service_level_pct,1)}% safety buffer`;
  }else{
    $('pp-safety').textContent=`0.00 ${unit}`;
    if($('pp-safety-note'))$('pp-safety-note').textContent='No service level given (0 safety buffer)';
    if($('pp-reorder-note'))$('pp-reorder-note').textContent='Deterministic lead-time demand (no buffer)';
  }
  $('pp-reorder').textContent=`${fmt(pp.reorder_point)} ${unit}`;
  $('pp-annual-cost').textContent=fmt(pp.annual_relevant_cost);
  if($('pp-cost-note')){
    $('pp-cost-note').textContent=`Order: ${fmt(pp.annual_ordering_cost)} + Cycle: ${fmt(pp.annual_cycle_holding)} + Buffer: ${fmt(pp.annual_buffer_holding)}`;
  }

  const box=$('pp-order-now-box');
  box.classList.toggle('order-urgent',Boolean(pp.order_now));
  const firstEvent=pp.order_schedule?.[0];
  const immediateQty=firstEvent?.Order_Date===pp.plan_start_date?(firstEvent.Total_Order_Quantity||pp.EOQ):pp.EOQ;
  $('pp-order-now-heading').textContent=pp.order_now?`Order now: place one order for ${fmt(immediateQty)} ${unit} on ${pp.plan_start_date}`:'No immediate order required today';
  $('pp-order-now-text').textContent=pp.order_now_explanation;

  const chips=$('pp-order-dates-chips');
  chips.replaceChildren();
  if(pp.expected_order_dates&&pp.expected_order_dates.length){
    pp.expected_order_dates.slice(0,5).forEach(d=>chips.append(el('span',d,'date-chip')));
    const more=pp.expected_order_dates.length-5;
    if(more>0)chips.append(el('span',`+${more} more`,'date-chip muted-chip'));
    $('pp-show-all-dates').hidden=pp.expected_order_dates.length<=5;
    $('pp-show-all-dates').onclick=()=>document.querySelector('.pp-schedule-section')?.scrollIntoView({behavior:'smooth',block:'start'});
  }else{
    chips.append(el('span','None projected during this forecast horizon','hint'));
    $('pp-show-all-dates').hidden=true;
  }

  const schedWrap=$('pp-schedule-table');
  schedWrap.replaceChildren();
  if(!pp.order_schedule||!pp.order_schedule.length){
    schedWrap.append(el('p',`Current stock stays above the reorder point (${fmt(pp.reorder_point)} ${unit}) across the entire forecast horizon.`,'empty'));
  }else{
    const table=el('table'),thead=el('thead'),trH=el('tr');
    ['Order #','Order date','EOQ batches',`Total quantity (${unit})`,`Inventory position (${unit})`,'Expected receipt',`Stock / backlog (${unit})`].forEach(h=>trH.append(el('th',h)));
    thead.append(trH);table.append(thead);
    const tbody=el('tbody');
    pp.order_schedule.forEach(r=>{
      const tr=el('tr');
      const stock=Number(r.Projected_Stock_At_Trigger||0);
      const backlog=Number(r.Backlog_At_Trigger||0);
      const stockText=backlog>0?`Backlog ${fmt(backlog)}`:`On hand ${fmt(Math.max(0,stock))}`;
      tr.append(
        el('td',`#${r.Order_Number}`),
        el('td',`${r.Order_Date} (${r.Order_Weekday.slice(0,3)})`),
        el('td',String(r.EOQ_Batch_Count||1)),
        el('td',fmt(r.Total_Order_Quantity||r.Order_Quantity_EOQ)),
        el('td',fmt(r.Inventory_Position_At_Trigger)),
        el('td',`${r.Expected_Receipt_Date} (${r.Receipt_Weekday.slice(0,3)})`),
        el('td',stockText,backlog>0?'backlog-cell':'')
      );
      tbody.append(tr);
    });
    table.append(tbody);
    schedWrap.append(table);
  }
  $('download-pp-schedule').onclick=()=>{
    download(new Blob([pp.order_schedule_csv||''],{type:'text/csv;charset=utf-8'}),'purchase_order_schedule.csv');
  };

  const ppGal=$('pp-gallery');
  ppGal.replaceChildren();
  state.result.figures.filter(f=>f.part===2).forEach(f=>{
    ppGal.append(buildGraphCard(f,()=>renderPurchasePlanSection()));
  });
}

function handlePurchaseChoice(wantsPlan){
  state.purchaseDecision=wantsPlan;
  if(!wantsPlan){
    $('pp-form').hidden=true;
    $('pp-results').hidden=true;
    if($('cta-no-msg'))$('cta-no-msg').hidden=false;
    if($('pp-declined-note'))$('pp-declined-note').hidden=false;
  }else{
    if($('cta-no-msg'))$('cta-no-msg').hidden=true;
    if($('pp-declined-note'))$('pp-declined-note').hidden=true;
    $('pp-form').hidden=false;
    if($('pp-stock-unit-hint'))$('pp-stock-unit-hint').textContent=`in ${activeUnitLabel()}`;
    view('purchase');
    updateStepper(5);
    $('pp-form').scrollIntoView({behavior:'smooth',block:'start'});
  }
}

$('cta-pp-yes')?.addEventListener('click',()=>handlePurchaseChoice(true));
$('cta-pp-no')?.addEventListener('click',()=>handlePurchaseChoice(false));
$('pp-yes-btn')?.addEventListener('click',()=>handlePurchaseChoice(true));
$('pp-no-btn')?.addEventListener('click',()=>handlePurchaseChoice(false));

function validatePurchaseForm(){
  const startDate=$('pp-start-date').value.trim();
  const curStockRaw=$('pp-current-stock').value.trim();
  const leadRaw=$('pp-lead-time').value.trim();
  const orderCostRaw=$('pp-order-cost').value.trim();
  const holdCostRaw=$('pp-holding-cost').value.trim();
  const serviceRaw=$('pp-service').value.trim();

  if(!startDate)return 'Please enter the plan start date (YYYY-MM-DD).';
  const dt=new Date(startDate+'T12:00:00Z');
  if(Number.isNaN(dt.getTime()))return 'Please enter a valid plan start date in YYYY-MM-DD format.';
  if(!usesAllCalendarDays()&&[0,6].includes(dt.getUTCDay()))return 'The plan start date falls on a weekend. Please choose a Monday–Friday operating day.';

  if(curStockRaw==='')return `Please enter your current stock on hand (in ${activeUnitLabel()}).`;
  const curStock=Number(curStockRaw);
  if(!Number.isFinite(curStock)||curStock<0)return 'Current stock on hand must be a number of 0 or greater.';

  const planDayLabel=usesAllCalendarDays()?'calendar days':'operating days';
  if(leadRaw==='')return `Please enter the supplier lead time in ${planDayLabel} (1 to 20).`;
  const lead=Number(leadRaw);
  if(!Number.isInteger(lead)||lead<1||lead>20)return `Supplier lead time must be a whole number of ${planDayLabel} between 1 and 20.`;
  if(lead>Number($('horizon').value))return `Supplier lead time (${lead} days) cannot exceed the forecast horizon (${$('horizon').value} days).`;

  if(orderCostRaw==='')return 'Please enter the cost per order.';
  const orderCost=Number(orderCostRaw);
  if(!Number.isFinite(orderCost)||orderCost<=0)return 'Cost per order must be a positive number greater than 0.';

  if(holdCostRaw==='')return 'Please enter the holding cost per unit per year.';
  const holdCost=Number(holdCostRaw);
  if(!Number.isFinite(holdCost)||holdCost<=0)return 'Holding cost per unit per year must be a positive number greater than 0.';

  if(serviceRaw!==''&&serviceRaw!=='0'){
    const service=Number(serviceRaw);
    if(!Number.isFinite(service)||service<51||service>99.9)return 'Target service level must be between 51% and 99.9% (for example, 95), or left blank for 0 safety stock.';
  }

  return null;
}

$('pp-form')?.addEventListener('submit',async e=>{
  e.preventDefault();
  const errEl=$('pp-error');
  const msg=validatePurchaseForm();
  if(msg){
    errEl.textContent=msg;
    errEl.hidden=false;
    return;
  }
  errEl.hidden=true;
  await runAnalysis(true);
});

const TABLE_PURPOSES={
  model_leaderboard:'One-glance ranking of all evaluated models sorted by validation MAE alongside final test accuracy, MASE, bias, and cross-validation MAE.',
  weekly_forecast_summary:'Weekly roll-up of projected demand across the selected operating calendar with 80% and 95% weekly bounds for capacity and procurement planning.',
  model_metrics:'Compare model accuracy on the untouched final test period across rolling 1-step, lead-time block, and fixed-horizon protocols.',
  interval_coverage:'Check how often actual values fell inside the forecast intervals. Coverage near the stated 80% or 95% level is desirable.',
  cv_summary:'Check whether model performance remains stable across multiple rolling-origin validation folds.',
  diebold_mariano:'Test whether the selected model’s errors differ meaningfully from the comparison model; p-values below 0.05 are stronger evidence.',
  feature_comparison:'Compare the simpler baseline feature set with the richer feature set on validation and final test data.',
  data_preparation_log:'See exactly which blank values or outliers were detected and what treatment, if any, was applied.',
  service_sensitivity:'Compare EOQ, safety stock, reorder point, and annual cost across No buffer (0%), 90%, 95%, and 99% service targets.'
};
function compactColumns(data){
  if(state.tableFull||data.columns.length<=8)return data.columns;
  const preferred=['Validation_Rank','Operating_Week','Step_Range','Model','Model_Label','Role','Validation_MAE','Test_MAE','MAE','RMSE','MASE','sMAPE_pct','WAPE_pct','Bias','Tracking_Signal','Weekly_Total','Daily_Mean','Fold','Mean_MAE','p_value','Verdict','Observation','Weekday_Name','Action','Forecast_Date','Date','Forecast_Step','Forecast'];
  const chosen=preferred.filter(c=>data.columns.includes(c)).slice(0,8);
  return chosen.length>=3?chosen:data.columns.slice(0,8);
}

function populateTableSelect(){
  const select=$('table-select');
  if(!select||!state.result)return;
  const prevVal=select.value;
  select.replaceChildren();
  const filtered=state.result.tables.map((t,i)=>({t,i})).filter(({t})=>state.tableCategory==='All'||tableCategoryFor(t.key)===state.tableCategory);
  filtered.forEach(({t,i})=>select.add(new Option(t.label,String(i))));
  if(filtered.some(x=>String(x.i)===prevVal)){
    select.value=prevVal;
  }else if(filtered.length){
    const def=filtered.find(x=>x.t.key==='model_leaderboard')||filtered.find(x=>x.t.key==='model_metrics')||filtered[0];
    select.value=String(def.i);
  }
}

function renderTable(){
  if(!state.result||!state.result.tables.length)return;
  const idx=Number($('table-select').value||0);
  const data=state.result.tables[idx]||state.result.tables[0];
  const container=$('results-table');container.replaceChildren();
  const purpose=TABLE_PURPOSES[data.key]||`This table contains ${data.rows.length.toLocaleString('en-US')} row${data.rows.length===1?'':'s'} of numerical results. Click any column header to sort, or download the CSV.`;
  $('table-context').hidden=false;$('table-context').replaceChildren(el('strong','What this table tells you'),el('p',purpose));
  const columns=compactColumns(data);
  $('table-glossary').hidden=false;
  const glossHolder=$('table-glossary-content');
  if(glossHolder){
    glossHolder.replaceChildren();
    const gGrid=el('div',undefined,'glossary-grid');
    columns.forEach(c=>{
      const desc=COLUMN_DEFINITIONS[c]||`Values for ${c.replaceAll('_',' ').toLowerCase()}.`;
      const chip=el('div',undefined,'glossary-chip');
      chip.append(el('strong',c.replaceAll('_',' ')+': '),el('span',desc));
      gGrid.append(chip);
    });
    glossHolder.append(gGrid);
  }
  $('table-detail-toggle').disabled=data.columns.length<=8;
  $('table-detail-toggle').textContent=state.tableFull?'Show compact view':'Show full details';

  const searchQ=(state.tableSearch||'').trim().toLowerCase();
  let rowsToRender=data.rows.filter(r=>!searchQ||columns.some(c=>String(r[c]??'').toLowerCase().includes(searchQ)));
  if(state.tableSortCol&&columns.includes(state.tableSortCol)){
    const sc=state.tableSortCol;
    const dir=state.tableSortAsc?1:-1;
    rowsToRender=[...rowsToRender].sort((a,b)=>{
      const va=a[sc],vb=b[sc];
      if(typeof va==='number'&&typeof vb==='number')return (va-vb)*dir;
      return String(va??'').localeCompare(String(vb??''))*dir;
    });
  }

  const table=el('table'),head=el('tr');
  columns.forEach(c=>{
    const arrow=state.tableSortCol===c?(state.tableSortAsc?' ▲':' ▼'):'';
    const th=el('th',c.replaceAll('_',' ')+arrow,'sortable-th');
    th.onclick=()=>{
      if(state.tableSortCol===c)state.tableSortAsc=!state.tableSortAsc;
      else{state.tableSortCol=c;state.tableSortAsc=true;}
      renderTable();
    };
    head.append(th);
  });
  const thead=el('thead');thead.append(head);table.append(thead);
  const body=el('tbody');
  const winModel=state.result.summary?.selected_model;
  rowsToRender.slice(state.tablePage*25,state.tablePage*25+25).forEach(row=>{
    const tr=el('tr');
    if(row.Model&&row.Model===winModel)tr.className='winner-row';
    columns.forEach(c=>tr.append(el('td',fmtCell(row[c],c))));
    body.append(tr);
  });
  table.append(body);container.append(table);
  if(!rowsToRender.length)container.append(el('p','No matching rows in this table.','empty'));
  paginate($('table-pages'),state.tablePage,rowsToRender.length,p=>{state.tablePage=p;renderTable();});
}
$('table-select').onchange=()=>{state.tablePage=0;state.tableFull=false;state.tableSortCol=null;renderTable();};
$('table-detail-toggle').onclick=()=>{state.tableFull=!state.tableFull;renderTable();};
$('table-row-search')?.addEventListener('input',e=>{state.tableSearch=e.target.value;state.tablePage=0;renderTable();});
$('download-table').onclick=()=>{const t=state.result.tables[Number($('table-select').value)];download(new Blob([t.csv],{type:'text/csv;charset=utf-8'}),t.key+'.csv');};
$('download-all').onclick=()=>download(fromBase64(state.result.zip,'application/zip'),'demand_forecast_results.zip');

function showResult(result,elapsed,fromPurchasePlan=false){
  state.result=result;state.runVersion=state.version;state.hasRun=true;
  if($('apply-data-edits'))$('apply-data-edits').hidden=true;
  state.selectedGalleryKeys.clear();
  state.selectedOverviewKeys.clear();
  result.figures.forEach(f=>{
    if(f.method_views)f.activeMethod=f.default_method||result.summary.selected_model||'SMA';
    state.selectedGalleryKeys.add(f.key);
    if(RECOMMENDED_GRAPH_KEYS.has(f.key)&&f.part!==2&&f.key!=='02_evaluation'){
      state.selectedOverviewKeys.add(f.key);
    }
  });
  const s=result.summary;
  const dtype=s.data_type||activeDataTypeLabel();
  const unit=s.unit_name||activeUnitLabel();
  const disp=displayLabelWithUnit(dtype,unit);
  if($('fast-mode')&&!state.fastModeUserTouched)$('fast-mode').checked=Boolean(s.fast_mode);
  $('initial').hidden=true;$('result-overview').hidden=false;$('dirty').hidden=true;
  if($('report-kicker'))$('report-kicker').textContent=`PART 1 · FORECAST REPORT FOR ${disp.toUpperCase()} — SELECTED ON VALIDATION`;
  $('selected-model').textContent=`${s.selected_label} (${disp})`;
  $('result-description').textContent=`${s.n.toLocaleString('en-US')} observations · ${s.split.train.toLocaleString('en-US')} training / ${s.split.validation.toLocaleString('en-US')} validation / ${s.split.test.toLocaleString('en-US')} final test. Validation MAE: ${fmt(s.selected_validation_MAE)} ${unit}.`;

  if($('fast-mode-banner')){
    if(s.fast_mode&&s.fast_mode_message){
      $('fast-mode-banner').textContent=s.fast_mode_message;
      $('fast-mode-banner').hidden=false;
    }else{
      $('fast-mode-banner').textContent='';
      $('fast-mode-banner').hidden=true;
    }
  }

  if($('model-warning')){
    if(s.unavailable_models&&s.unavailable_models.length){
      const uniqReasons=[...new Set(s.unavailable_models.map(u=>`${u.label}: ${u.reason}`))];
      $('model-warning').textContent='Warning — '+uniqReasons.join(' ');
      $('model-warning').hidden=false;
    }else{
      $('model-warning').textContent='';
      $('model-warning').hidden=true;
    }
  }

  if($('cv-warning')){
    if(s.cv_warning){
      $('cv-warning').textContent=s.cv_warning;
      $('cv-warning').hidden=false;
    }else{
      $('cv-warning').textContent='';
      $('cv-warning').hidden=true;
    }
  }

  if($('model-comparison-note')){
    const evalNames=(s.models_evaluated||[]).map(m=>s.model_labels?.[m]||m).join(', ');
    const beatText=s.new_models_beat_baseline?'One of the newly added models improved validation accuracy over the original 4 baseline methods.':'Honest validation check: none of the newly added models had a lower validation MAE than the winning baseline method on this dataset.';
    const cvText=s.cv_winner?` Rolling-origin ${s.cv_folds||5}-fold CV lowest mean MAE: ${s.cv_winner_label||s.cv_winner}.`:'';
    $('model-comparison-note').textContent=`Compared ${s.models_evaluated?.length||10} models (${evalNames}). ${beatText}${cvText}`;
  }

  renderExecutiveSummary(result);

  $('run-time').textContent=`Completed in ${elapsed.toFixed(2)} s${s.fast_mode?' (Fast mode)':''}`;
  $('metric-mae').textContent=`${fmt(s.test_metrics.MAE)} ${unit}`;
  $('metric-rmse').textContent=`${fmt(s.test_metrics.RMSE)} ${unit}`;
  $('metric-wape').textContent=`${fmt(s.test_metrics.WAPE_pct)}%`;
  $('metric-bias').textContent=`${fmt(s.test_metrics.Bias)} ${unit}`;

  const wapeV=Number(s.test_metrics.WAPE_pct||0);
  if($('badge-wape')){
    $('badge-wape').textContent=wapeV<10?'High accuracy':(wapeV<20?'Good':'Moderate');
    $('badge-wape').className='metric-pill '+(wapeV<10?'good':(wapeV<20?'neutral':'warn'));
  }
  if($('badge-mae')){$('badge-mae').textContent='Primary error';$('badge-mae').className='metric-pill neutral';}
  if($('badge-rmse')){$('badge-rmse').textContent='Large-error sensitivity';$('badge-rmse').className='metric-pill neutral';}

  const biasV=Number(s.test_metrics.Bias||0);
  const tsV=Number(s.test_metrics.Tracking_Signal||0);
  if($('badge-bias')){
    $('badge-bias').textContent=Math.abs(biasV)<1e-6?'Unbiased':(biasV>0?'Over-forecasts':'Under-forecasts');
    $('badge-bias').className='metric-pill '+(Math.abs(tsV)<=4?'good':'warn');
  }
  if($('metric-bias-note')){
    $('metric-bias-note').textContent=`Tracking signal: ${fmt(tsV,2)} (${Math.abs(tsV)<=4?'within ±4 control limit':'drift alert > ±4'})`;
  }

  if($('metric-mase')){
    $('metric-mase').textContent=fmt(s.test_metrics.MASE,3);
    const maseVal=s.test_metrics.MASE;
    if($('badge-mase')){
      $('badge-mase').textContent=maseVal<1?'Beats Naive':'Above Naive';
      $('badge-mase').className='metric-pill '+(maseVal<1?'good':'warn');
    }
    if($('metric-mase-note')){
      $('metric-mase-note').textContent=maseVal<1?`Below 1.00: beats 1-step naive (scale ${fmt(s.naive_scale_train,2)} ${unit})`:`Above 1.00 vs 1-step naive (scale ${fmt(s.naive_scale_train,2)} ${unit})`;
    }
  }
  if($('metric-smape')){
    $('metric-smape').textContent=`${fmt(s.test_metrics.sMAPE_pct)}%`;
    if($('badge-smape')){$('badge-smape').textContent='Symmetric %';$('badge-smape').className='metric-pill neutral';}
    if($('metric-smape-note')){
      const zEx=s.test_metrics.sMAPE_zero_excluded||0;
      $('metric-smape-note').textContent=`Symmetric % error (${zEx} zero pair${zEx===1?'':'s'} excluded)`;
    }
  }

  if($('interval-horizon-note')){
    const maxSup=s.interval_max_supported_horizon??0;
    const minErrRule=s.interval_min_errors_rule??30;
    const hz=s.parameters?.horizon||60;
    let hzMsg='';
    if(maxSup>=hz){
      hzMsg=`Prediction intervals (80% & 95%): Shown across all ${hz} future steps (each step has ≥ ${minErrRule} rolling backtest errors).`;
    }else if(maxSup>0){
      hzMsg=`Prediction intervals (80% & 95%): Shown for future steps 1 to ${maxSup} (≥ ${minErrRule} backtest errors). Steps ${maxSup+1} to ${hz} have fewer than ${minErrRule} errors.`;
    }else{
      hzMsg=`Prediction intervals (80% & 95%): Omitted on future steps because fewer than ${minErrRule} backtest errors are available (adjust in Step 2 Advanced Settings if desired).`;
    }
    const covTable=result.tables.find(t=>t.key==='interval_coverage');
    const winCov=covTable?.rows?.find(r=>r.Model===s.selected_model);
    const covMsg=winCov?` Test coverage for ${s.selected_label}: ${fmt(winCov.Empirical_80_Coverage_pct,1)}% inside 80% band and ${fmt(winCov.Empirical_95_Coverage_pct,1)}% inside 95% band.`:'';
    $('interval-horizon-note').textContent=hzMsg+covMsg;
  }

  renderLeaderboard(result);
  renderSummaryPanelsAndInlineTables(result);
  syncFeatured();
  renderOverviewCards();

  $('warnings').replaceChildren(...s.warnings.map(w=>el('li',w)));
  $('saved-path').textContent=result.saved_to?'Saved on this computer: '+result.saved_to:'This browser run is temporary. Download the ZIP to keep every chart and CSV table.';

  $('graph-filters').replaceChildren();
  state.graphGroup='All';
  ['All','Exploration','Data','Forecasts','Evaluation','Diagnostics','Inventory'].forEach(group=>{
    const b=el('button',group,group==='All'?'active':'');
    b.type='button';
    b.onclick=()=>{
      state.graphGroup=group;
      [...$('graph-filters').children].forEach(x=>x.classList.toggle('active',x===b));
      const pool=getGalleryCategoryPool();
      if(!pool.some(f=>state.selectedGalleryKeys.has(f.key))){
        pool.forEach(f=>state.selectedGalleryKeys.add(f.key));
      }
      renderGallery();
    };
    $('graph-filters').append(b);
  });
  renderGallery();

  const catPills=$('table-category-pills');
  if(catPills){
    catPills.replaceChildren();
    state.tableCategory='All';
    Object.entries(TABLE_CATEGORIES).forEach(([catKey,catLabel])=>{
      const btn=el('button',catLabel,catKey==='All'?'active':'');
      btn.type='button';
      btn.onclick=()=>{
        state.tableCategory=catKey;
        [...catPills.children].forEach(x=>x.classList.toggle('active',x===btn));
        state.tablePage=0;state.tableSortCol=null;
        populateTableSelect();
        renderTable();
      };
      catPills.append(btn);
    });
  }
  populateTableSelect();
  $('table-select').disabled=false;
  if($('table-row-search'))$('table-row-search').disabled=false;
  $('download-table').disabled=false;
  $('table-detail-toggle').disabled=false;
  state.tablePage=0;state.tableFull=false;
  renderTable();

  $('download-all').disabled=false;
  const activeGalleryCount=getGalleryCategoryPool().length;
  document.querySelector('[data-view="graphs"]').textContent=`All Graphs (${activeGalleryCount})`;

  if(fromPurchasePlan&&result.purchase_plan){
    renderPurchasePlanSection();
    view('purchase');
    updateStepper(5);
    $('pp-results').scrollIntoView({behavior:'smooth',block:'start'});
  }else{
    renderPurchasePlanSection();
    updateStepper(4);
    const activeTab=document.querySelector('[data-view].active')?.dataset?.view;
    if(!activeTab||activeTab==='overview'){
      view('overview');
      if(innerWidth<=720)$('overview').scrollIntoView({behavior:'smooth',block:'start'});
    }
  }
}

function setBusy(busy){
  state.busy=busy;
  $('run').disabled=busy||!state.rows.length;
  $('run').classList.toggle('busy',busy);
  $('run').firstChild.textContent=busy?'Generating forecast report ':'Generate Part 1 Forecast Report ';
  $('cancel').hidden=!busy||state.mode!=='browser';
  for(const id of ['file','sample','demand-column','calendar-column','calendar-type','calendar-mode','start-weekday','data-type','custom-data-type','unit-name','horizon','cv-folds','min-interval-errors','missing-treatment','max-missing-pct','outlier-treatment','outlier-threshold','hw-seasonal','hw-damped','fast-mode','fast-mode-threshold','add-row']){
    if($(id))$(id).disabled=busy;
  }
  if($('pp-submit'))$('pp-submit').disabled=busy;
  if(!busy){
    $('calendar-type').disabled=$('calendar-column').value==='';
    $('start-weekday').disabled=$('calendar-column').value!=='';
  }
}

function payload(includePurchasePlan=false){
  const settings={
    data_type:$('data-type').value,
    custom_data_type:$('custom-data-type').value.trim(),
    unit_name:activeUnitLabel(),
    horizon:$('horizon').value,
    cv_folds:$('cv-folds')?$('cv-folds').value:5,
    min_interval_errors:$('min-interval-errors')?$('min-interval-errors').value:30,
    missing_treatment:$('missing-treatment')?$('missing-treatment').value:'interpolate',
    max_missing_pct:$('max-missing-pct')?$('max-missing-pct').value:10,
    outlier_treatment:$('outlier-treatment')?$('outlier-treatment').value:'keep',
    outlier_threshold:$('outlier-threshold')?$('outlier-threshold').value:3.2,
    hw_seasonal:$('hw-seasonal')?$('hw-seasonal').value:'auto',
    hw_damped:$('hw-damped')?$('hw-damped').value:'auto',
    fast_mode:$('fast-mode')?Boolean($('fast-mode').checked):false,
    fast_mode_row_threshold:$('fast-mode-threshold')?$('fast-mode-threshold').value:500,
    start_weekday:$('start-weekday').value,
    calendar_mode:$('calendar-mode')?.value||'weekdays',
    operating_days_year:usesAllCalendarDays()?365:260,
    purchase_plan_requested:Boolean(includePurchasePlan)
  };
  if(includePurchasePlan){
    settings.plan_start_date=$('pp-start-date').value.trim();
    settings.current_stock=$('pp-current-stock').value.trim();
    settings.lead_time=$('pp-lead-time').value.trim();
    settings.order_cost=$('pp-order-cost').value.trim();
    settings.holding_cost_unit_year=$('pp-holding-cost').value.trim();
    settings.service_level=$('pp-service').value.trim();
  }
  return {rows:structuredClone(state.rows),settings};
}

function browserRun(input){
  return new Promise((resolve,reject)=>{
    state.worker ||= new Worker('worker.mjs',{type:'module'});
    state.rejectRun=reject;
    state.worker.onmessage=({data})=>{
      if(data.type==='progress'){
        $('status').textContent=data.message;
        updateProgressUI(data.message);
      }
      if(data.type==='result')resolve(data.result);
      if(data.type==='error')reject(new Error(data.message));
    };
    state.worker.onerror=()=>{
      state.worker?.terminate();state.worker=null;
      reject(new Error('The browser Python runtime could not load. Check your internet connection and refresh the page.'));
    };
    state.worker.postMessage({id:Date.now(),payload:input});
  });
}

$('cancel').onclick=()=>{
  state.pendingRun=false;
  state.worker?.terminate();
  state.worker=null;
  updateProgressUI(null);
  state.rejectRun?.(new Error('Run cancelled. You can change the inputs and run again.'));
};

async function runAnalysis(includePurchasePlan=false){
  if($('data-type').value==='Custom'&&!$('custom-data-type').value.trim()){
    setSettingsCollapsed(false);
    error('Please enter a custom data type name in Step 2 on the left.');
    $('custom-data-type').focus();
    return;
  }
  if(!$('unit-name').value.trim()){
    setSettingsCollapsed(false);
    error('Please enter a unit name in Step 2 on the left (for example: units, kWh, orders).');
    $('unit-name').focus();
    return;
  }
  if(state.busy){state.pendingRun=true;return;}
  if(state.editTimer){clearTimeout(state.editTimer);state.editTimer=null;}
  state.pendingRun=false;
  error('');
  const start=performance.now();
  setBusy(true);
  updateStepper(3);
  const disp=displayLabelWithUnit().toLowerCase();
  const startMsg=state.mode==='local'
    ?`Step 1/8: Validating and preparing ${disp}…`
    :`Starting browser Python for ${disp}…`;
  $('status').textContent=startMsg;
  updateProgressUI(startMsg,12);
  try{
    const input=payload(includePurchasePlan);
    let result;
    if(state.mode==='local'){
      const response=await fetch('api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
      result=await response.json();
      if(!response.ok)throw new Error(result.error||'The analysis did not finish.');
    }else result=await browserRun(input);
    updateProgressUI('Step 8/8: Analysis complete',100);
    showResult(result,(performance.now()-start)/1000,includePurchasePlan); enhancePresentation(result);
    $('status').textContent=chartCountText(result);
  }catch(e){
    error(e.message);
    updateProgressUI(null);
    if(includePurchasePlan&&$('pp-error')){
      $('pp-error').textContent=e.message;
      $('pp-error').hidden=false;
    }
    $('status').textContent='Check the message above and try again.';
  }finally{
    setBusy(false);
    if(state.pendingRun){
      state.pendingRun=false;
      runAnalysis(false);
    }
  }
}

$('run').onclick=()=>runAnalysis(false);

async function init(){
  await sample(false);
  if(new URLSearchParams(location.search).get('engine')!=='browser'){
    try{
      const r=await fetch('api/health',{signal:AbortSignal.timeout(1800)});
      if(r.ok&&(await r.json()).engine==='local-python')state.mode='local';
    }catch{}
  }
  $('status').textContent=state.mode==='local'?'Ready. Confirm your data type and unit above, then click Generate Part 1 Forecast Report.':'Ready. Browser Python loads on the first run; confirm your data type and unit above.';
}
init();

// Presentation enhancements consume result values only.
const KEY_CHARTS=new Set(['01_demand','02_evaluation','03_future','06_errors','14_cv_folds','10_policy','11_cost']);
function chartAlt(f,v){return `${v.title} - ${state.result?.summary.selected_label||'forecast model'} - ${f.chart_type||'chart'}`;}
function chartCountText(r){const n=r.figures.filter(f=>f.part!==2||r.purchase_plan).length;return r.purchase_plan?`${n} charts ready`:`${n} charts ready (Part 2 adds 3 inventory charts)`;}
function enhancePresentation(r){
 const s=r.summary,e=s.executive_summary,u=s.unit_name;
 const dayLabel=s.parameters?.calendar_mode==='all_days'?'calendar days':'operating days';
 const answer=$('executive-summary-card');answer.replaceChildren(el('span','FORECAST ANSWER','eyebrow'),el('h2',s.selected_label),el('p',`Next 5 ${dayLabel}: ${fmt(e.next_5_days_total)} ${u} total · ${fmt(e.next_5_days_mean)} ${u}/day. Typical final-test error: ${fmt(s.test_metrics.MAE)} ${u}.`));
 if(!$('more-metrics')){const d=el('details');d.id='more-metrics';d.append(el('summary','More metrics'));const grid=el('div',undefined,'metrics');for(const id of ['metric-rmse','metric-bias','metric-mase','metric-smape'])grid.append($(id).closest('.metric-card'));d.append(grid);document.querySelector('#result-overview .metrics').after(d);}
 for(const id of ['primary-next5','primary-eoq'])$(id)?.remove();
 const grid=document.querySelector('#result-overview .metrics');
 const add=(id,label,value)=>{const x=el('div',undefined,'metric-card');x.id=id;x.append(el('span',label),el('strong',value));grid.append(x);};
 add('primary-next5',`Next 5 ${dayLabel}`,`${fmt(e.next_5_days_total)} ${u}`);if(r.purchase_plan)add('primary-eoq','Base EOQ',`${fmt(r.purchase_plan.EOQ)} ${u}`);
 if(!$('secondary-report')){const d=el('details');d.id='secondary-report';d.append(el('summary','Model leaderboard and analysis details'));for(const id of ['leaderboard-section','exploration-prep-panels','inline-comparison-tables'])if($(id))d.append($(id));$('more-metrics').after(d);}
 if(!$('report-notes')){const d=el('details');d.id='report-notes';d.append(el('summary','Notes'));for(const id of ['model-warning','cv-warning','model-comparison-note','interval-note'])if($(id))d.append($(id));$('more-metrics').after(d);}
 $('result-overview').prepend(answer); for(const id of ['fast-mode-banner','interval-horizon-note'])if($(id))$('report-notes').append($(id)); const nd=$('report-notes');nd.querySelector('summary').textContent=`Notes (${s.warnings?.length||0})`;
 $('graph-search-message').textContent=chartCountText(r);
 const pp=r.purchase_plan;if(pp){const first=pp.order_schedule?.[0];$('pp-order-now-heading').textContent=pp.order_now?`Order now: Yes · ${fmt(first?.Total_Order_Quantity||pp.EOQ)} ${u} · arrives ${first?.Expected_Receipt_Date||'—'}`:'Order now: No';
 let note=$('pp-shortage');if(!note){note=el('p');note.id='pp-shortage';$('pp-order-now-box').append(note);}const backlog=pp.order_schedule.find(x=>x.Backlog_At_Trigger>0);note.textContent=backlog?`Starting stock is below the reorder point; the plan shows a shortage of ${fmt(backlog.Backlog_At_Trigger)} ${u} on ${backlog.Order_Date}.`:'';
 let fast=$('pp-fast-note');if(!fast){fast=el('p',undefined,'hint');fast.id='pp-fast-note';$('pp-order-now-box').append(fast);}fast.textContent=s.fast_mode?'Fast mode skipped model tuning, so EOQ and ROP can differ slightly from a full run.':'';
 $('pp-results').prepend($('pp-order-now-box'));$('pp-order-now-box').after(document.querySelector('.pp-schedule-section'));let editor=$('pp-input-editor');if(!editor){editor=el('details');editor.id='pp-input-editor';editor.append(el('summary','Edit planning inputs'));$('pp-form').before(editor);editor.append($('pp-form'));}editor.open=false;editor.before($('pp-results'));}
 if(!pp&&$('pp-input-editor'))$('pp-input-editor').open=true;
 renderTableCatalog(r);
}
function renderTableCatalog(r){
 let catalog=$('table-catalog');if(!catalog){catalog=el('div');catalog.id='table-catalog';$('tables').append(catalog);}catalog.replaceChildren();
 const important=new Set(['model_leaderboard','forecast_5_steps','weekly_forecast_summary','inventory_metrics','cv_summary']);
 for(const t of r.tables.filter(t=>(state.tableCategory==='All'||tableCategoryFor(t.key)===state.tableCategory)&&(!state.catalogSearch||t.label.toLowerCase().includes(state.catalogSearch)))){const d=el('details',undefined,'table-entry');d.open=important.has(t.key);d.append(el('summary',`${t.label} · ${t.rows.length.toLocaleString()} rows`));const draw=()=>{if(d.childElementCount>1)return;const wrap=el('div',undefined,'table-scroll');const tb=el('table'),head=el('tr');t.columns.forEach(c=>head.append(el('th',c.replaceAll('_',' '))));tb.append(head);t.rows.slice(0,25).forEach(row=>{const tr=el('tr');t.columns.forEach(c=>tr.append(el('td',fmtCell(row[c],c))));tb.append(tr);});wrap.append(tb);d.append(wrap,el('p',t.rows.length>25?'Preview: first 25 rows. Use the sortable table viewer or CSV for all rows.':''));};d.ontoggle=()=>{if(d.open)draw();};if(d.open)draw();catalog.append(d);}
}
const showAll=el('button','Show all charts','secondary');showAll.id='show-all-charts';showAll.onclick=()=>{state.galleryAll=!state.galleryAll;showAll.textContent=state.galleryAll?'Show key charts':'Show all charts';showAll.setAttribute('aria-pressed',String(state.galleryAll));renderGallery();};$('graph-search').parentElement.before(showAll);
for(const [label,open] of [['Expand all guides',true],['Collapse all guides',false]]){const b=el('button',label,'secondary');b.onclick=()=>document.querySelectorAll('.guide-part').forEach(d=>d.open=open);showAll.after(b);}
const clear=el('button','Clear data','secondary');clear.id='clear-data';clear.onclick=()=>{const dialog=el('dialog');dialog.className='clear-confirm';dialog.setAttribute('aria-label','Clear loaded data');dialog.append(el('h2','Clear all loaded observations?'));const cancel=el('button','Keep data','secondary'),yes=el('button','Clear observations','primary');cancel.onclick=()=>{dialog.close();dialog.remove();};yes.onclick=()=>{state.rows=[];state.raw=[];state.name='';state.result=null;state.hasRun=false;state.pendingRun=false;if(state.editTimer){clearTimeout(state.editTimer);state.editTimer=null;}clearResults();renderInput();drawPreview();$('file').value='';$('file-info').textContent='No data loaded.';$('run').disabled=true;$('status').textContent='No data loaded. Load a file or the sample dataset.';dialog.close();dialog.remove();};dialog.append(cancel,yes);document.body.append(dialog);dialog.showModal();};$('download-input').after(clear);
document.querySelector('[role=tablist]').addEventListener('keydown',event=>{const tabs=[...document.querySelectorAll('[role=tab]')].filter(t=>!t.hidden);const i=tabs.indexOf(document.activeElement);if(i<0)return;let next;if(event.key==='ArrowRight')next=(i+1)%tabs.length;else if(event.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();view(tabs[next].dataset.view);tabs[next].focus();});
$('status').setAttribute('aria-live','polite');document.querySelectorAll('[role=tab]').forEach(t=>t.tabIndex=t.getAttribute('aria-selected')==='true'?0:-1);
const image=$('dialog-image'),viewport=el('div',undefined,'chart-viewport');image.before(viewport);viewport.append(image);const full=el('a','Open full size');full.id='open-full-size';full.target='_blank';full.rel='noopener';$('dialog-download').after(full);
let zoom=1,panX=0,panY=0,points=new Map(),lastDistance=0;
function applyChartZoom(){image.style.transform=`translate(${panX}px,${panY}px) scale(${zoom})`;}
function resetChartZoom(){zoom=1;panX=panY=0;points.clear();applyChartZoom();}
viewport.addEventListener('dblclick',()=>{zoom=zoom===1?2:1;panX=panY=0;applyChartZoom();});
viewport.addEventListener('pointerdown',e=>{viewport.setPointerCapture(e.pointerId);points.set(e.pointerId,[e.clientX,e.clientY]);lastDistance=0;});
viewport.addEventListener('pointermove',e=>{if(!points.has(e.pointerId))return;const old=points.get(e.pointerId);points.set(e.pointerId,[e.clientX,e.clientY]);if(points.size===2){const [a,b]=[...points.values()];const dist=Math.hypot(a[0]-b[0],a[1]-b[1]);if(lastDistance)zoom=Math.max(1,Math.min(6,zoom*dist/lastDistance));lastDistance=dist;}else if(zoom>1){panX+=e.clientX-old[0];panY+=e.clientY-old[1];}applyChartZoom();});
for(const event of ['pointerup','pointercancel'])viewport.addEventListener(event,e=>{points.delete(e.pointerId);lastDistance=0;});
if(matchMedia('(max-width: 800px)').matches)setSettingsCollapsed(true);

const optional=$('pp-service').parentElement;optional.classList.add('optional-input');const optHeading=el('h3','Optional safety-stock buffer');optional.prepend(optHeading);document.querySelector('.pp-grid').after(optional);
const catalogSearch=el('input');catalogSearch.type='search';catalogSearch.placeholder='Search table names';catalogSearch.setAttribute('aria-label','Search table names');catalogSearch.oninput=()=>{state.catalogSearch=catalogSearch.value.trim().toLowerCase();if(state.result)renderTableCatalog(state.result);};$('tables').querySelector('.result-heading').after(catalogSearch);
const sortable=el('details');sortable.id='sortable-viewer';sortable.append(el('summary','Search rows, sort columns and download CSV'));for(const id of ['table-controls','table-context','table-glossary','results-table','table-pages']){const x=$(id)||document.querySelector('.'+id);if(x)sortable.append(x);}$('tables').append(sortable);
$('table-category-pills').addEventListener('click',()=>{if(state.result)renderTableCatalog(state.result);});
