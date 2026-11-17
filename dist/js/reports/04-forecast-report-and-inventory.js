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
