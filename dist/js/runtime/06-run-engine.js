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
      reject(new Error('The browser Python runtime could not load. Check your internet connection, or use START_WEBSITE.cmd on your computer.'));
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

