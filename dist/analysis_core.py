"""Reproducible forecasting and equivalent-stock scenario. Run src/run_project.py."""
from pathlib import Path
from statistics import NormalDist
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t, chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/Daily_Demand_Forecasting_Orders.csv'
NAMES = ['Week_of_month', 'Day_of_week', 'Non_urgent_order', 'Urgent_order',
         'Order_type_A', 'Order_type_B', 'Order_type_C', 'Fiscal_sector_orders',
         'Traffic_controller_orders', 'Banking_orders_1', 'Banking_orders_2',
         'Banking_orders_3', 'Total_Demand']
MODELS = ['SES', 'SMA', 'Weekday_DR', 'Weekday_Naive']
PARAMS = dict(train_end=24, validation_end=40, horizon=60, lead_time=5,
              operating_days_year=260, service_level=.95, order_cost=50.,
              holding_cost_unit_year=2.)
WEEK_LENGTH = 5
DAY_CODES = tuple(range(2, 7))
WEEKDAY_NAMES = {2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday', 8: 'Sunday'}


def load_data(path=RAW):
    data = pd.read_csv(path, sep=';')
    if data.shape != (60, 13) or not data.columns[0].startswith('Week of the month'):
        raise ValueError('Expected the official 60-row 13-column UCI CSV')
    data.columns = NAMES
    data = data.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy()).all() or (data < 0).any().any():
        raise ValueError('Missing, nonfinite, or negative data')
    if not data.Day_of_week.isin(range(2, 7)).all():
        raise ValueError('Invalid weekday code')
    data.insert(0, 'Observation', np.arange(1, 61))
    return data


def next_weekday(day):
    return (int(day) - 1) % WEEK_LENGTH + 2


def consecutive(previous, current):
    return int(current) == next_weekday(previous)


def fit_ses(values):
    """Profile SSE over alpha; solve the optimal initial level analytically."""
    y = np.asarray(values, dtype=float)
    def profile(alpha):
        weight = (1 - alpha) ** np.arange(len(y))
        part = np.zeros(len(y))
        for i in range(1, len(y)):
            part[i] = alpha * y[i-1] + (1-alpha) * part[i-1]
        initial = max(0., float(weight @ (y-part) / (weight @ weight)))
        errors = y - (weight * initial + part)
        return float(errors @ errors), initial
    grid = np.linspace(0, 1, 101)
    index = min(range(len(grid)), key=lambda i: profile(grid[i])[0])
    lo, hi = grid[max(0, index-1)], grid[min(100, index+1)]
    result = minimize_scalar(lambda a: profile(a)[0], bounds=(lo, hi),
                             method='bounded', options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError('SES parameter estimation failed')
    alpha = min([0., 1., float(result.x)], key=lambda a: profile(a)[0])
    sse, initial = profile(alpha)
    level = initial
    for actual in y:
        level = alpha * actual + (1-alpha) * level
    return dict(alpha=alpha, initial=initial, level=float(level), SSE=sse)


def feature(day, lag):
    return np.array([1., *(float(day == d) for d in DAY_CODES[:-1]), float(lag)])


def regression_fit(y, days):
    # Exclude rows with skipped weekdays from lag-one parameter estimation.
    keep = [i for i in range(1, len(y)) if consecutive(days[i-1], days[i])]
    X = np.vstack([feature(days[i], y[i-1]) for i in keep])
    target = np.asarray(y)[keep]
    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    return beta, X, target, np.asarray(keep)


def predict(model, y, days, future_days, window=5):
    """No future targets enter any fit or recursion."""
    history = list(np.asarray(y, dtype=float))
    if model == 'SES':
        return np.full(len(future_days), fit_ses(history)['level'])
    beta = regression_fit(np.asarray(history), days)[0] if model == 'Weekday_DR' else None
    previous_day, previous_value = int(days[-1]), history[-1]
    weekday_history = {d: history[np.flatnonzero(np.asarray(days) == d)[-1]]
                       for d in DAY_CODES}
    output = []
    for day in future_days:
        day = int(day)
        if model == 'SMA':
            value = float(np.mean(history[-window:]))
        elif model == 'Weekday_Naive':
            value = weekday_history[day]
        elif model == 'Weekday_DR':
            # Bridge the minimum weekday-consistent gap using predictions only.
            count = (day - previous_day) % WEEK_LENGTH or WEEK_LENGTH
            for _ in range(count):
                previous_day = next_weekday(previous_day)
                previous_value = max(0., float(feature(previous_day, previous_value) @ beta))
            value = previous_value
        else:
            raise ValueError('Unknown model')
        output.append(value)
        history.append(value)
        weekday_history[day] = value
    return np.asarray(output)


def score(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError('Metric arrays must be nonempty and have equal shape')
    error = actual - forecast
    absolute = np.abs(error)
    denom = np.abs(actual).sum()
    return dict(N=len(actual), MAE=float(absolute.mean()),
                RMSE=float(np.sqrt(np.mean(error**2))),
                WAPE_pct=float(100*absolute.sum()/denom) if denom else np.nan,
                Bias=float(-error.mean()), Median_AE=float(np.median(absolute)),
                Q1_AE=float(np.quantile(absolute,.25)), Q3_AE=float(np.quantile(absolute,.75)),
                SD_AE=float(np.std(absolute,ddof=1)) if len(actual)>1 else 0.)


def choose_window(y, days):
    # SMA window selection uses training records 1-24 only.
    rows=[]
    for window in [3,5,7,10]:
        forecasts=[predict('SMA',y[:i],days[:i],[days[i]],window)[0]
                   for i in range(12,PARAMS['train_end'])]
        rows.append(dict(Window=window, **score(y[12:PARAMS['train_end']],forecasts)))
    table=pd.DataFrame(rows).sort_values(['MAE','Window'])
    return int(table.iloc[0].Window),table


def choose_model(y, days, window):
    # Only records 25-40 select the model. Records 41-60 are final test data.
    a,b=PARAMS['train_end'],PARAMS['validation_end']
    rows=[]
    for model in MODELS:
        pred=[predict(model,y[:i],days[:i],[days[i]],window)[0] for i in range(a,b)]
        rows.append(dict(Model=model,**score(y[a:b],pred)))
    table=pd.DataFrame(rows).sort_values(['MAE','Model'])
    return str(table.iloc[0].Model),table


def lead_calibration(y, days, model, window, lead=5):
    """Development-only cumulative errors; exclude windows crossing weekday gaps."""
    rows=[]
    for origin in range(PARAMS['train_end'],PARAMS['validation_end']-lead+1):
        segment=days[origin-1:origin+lead]
        if not all(consecutive(a,b) for a,b in zip(segment[:-1],segment[1:])):
            continue
        predicted=predict(model,y[:origin],days[:origin],days[origin:origin+lead],window)
        actual=float(np.sum(y[origin:origin+lead]))
        rows.append(dict(Origin=origin,End=origin+lead,Actual_Total=actual,
                         Forecast_Total=float(predicted.sum()),Error=actual-float(predicted.sum())))
    frame=pd.DataFrame(rows)
    if len(frame)<3:
        raise ValueError('Too few valid lead-time calibration windows')
    return frame


def inventory(mean, predicted_lead, lead_errors, service=.95):
    annual=float(mean)*PARAMS['operating_days_year']
    sigma=float(np.std(lead_errors,ddof=1))
    bias=float(np.mean(lead_errors))  # Actual minus predicted cumulative demand.
    if service is None or (isinstance(service, (int, float, np.floating)) and float(service) <= 0.):
        z=0.0
        safety=0.0
    else:
        z=NormalDist().inv_cdf(float(service))
        safety=z*sigma
    expected=max(0.,float(predicted_lead)+bias)
    q=float(np.sqrt(2*annual*PARAMS['order_cost']/PARAMS['holding_cost_unit_year']))
    ordering=annual/q*PARAMS['order_cost']
    cycle=q/2*PARAMS['holding_cost_unit_year']
    buffer=safety*PARAMS['holding_cost_unit_year']
    return dict(mean_daily=float(mean),annual_demand=annual,lead_forecast=float(predicted_lead),
                lead_bias=bias,lead_error_SD=sigma,z=z,safety_stock=safety,
                reorder_point=expected+safety,EOQ=q,annual_ordering_cost=ordering,
                annual_cycle_holding=cycle,annual_buffer_holding=buffer,
                annual_relevant_cost=ordering+cycle+buffer,
                service_level_used=None if (service is None or float(service)<=0.) else float(service))


def simulate_policy(demand,q,rop,lead,initial=None):
    """Continuous review with uniform demand within each operating day and backlog."""
    initial=float(rop+q if initial is None else initial)
    net,position,clock=initial,initial,0.
    pipeline=[];orders=0;receipts=0;cycle_ok=True;completed=[];history=[];events=[]
    area=0.;total_short=0.
    for day,rate in enumerate(np.asarray(demand,float)):
        day_orders=0;day_receipts=0;day_short=0.
        while clock < day+1-1e-10:
            while pipeline and pipeline[0][0] <= clock+1e-9:
                due,qty=pipeline.pop(0);net+=qty;receipts+=1;day_receipts+=qty
                if receipts>1:completed.append(cycle_ok)
                cycle_ok=True
                events.append(dict(Time=clock,Event='Receipt',Quantity=qty,
                                   Net_Inventory=net,Inventory_Position=position,
                                   Outstanding_Receipts=float(sum(x[1] for x in pipeline))))
            if position <= rop+1e-8:
                trigger_position=position
                trigger_net=net
                orders+=1;day_orders+=q;position+=q
                pipeline.append((clock+lead,q));pipeline.sort()
                events.append(dict(Time=clock,Event='Order',Quantity=q,
                                   Net_Inventory=trigger_net,Inventory_Position=trigger_position,
                                   Outstanding_Receipts=float(sum(x[1] for x in pipeline))))
                continue
            due=pipeline[0][0] if pipeline else np.inf
            crossing=clock+(position-rop)/rate if rate>0 else np.inf
            stop=min(day+1.,due,crossing)
            dt=max(0.,stop-clock)
            if dt<1e-12:
                raise RuntimeError('Policy event simulation stalled')
            consumption=rate*dt
            before=net;after=net-consumption
            filled=min(consumption,max(before,0.))
            shortage=consumption-filled
            day_short+=shortage
            if shortage>1e-8:cycle_ok=False
            positive_dt=min(dt,max(before,0.)/rate) if rate>0 else dt
            area+=max(before,0.)*positive_dt-rate*positive_dt**2/2
            net=after;position-=consumption;clock=stop
        total_short+=day_short
        history.append(dict(Step=day+1,Demand=rate,Net_Inventory=net,On_Hand=max(0.,net),
                            Backlog=max(0.,-net),Ordered=day_orders,Received=day_receipts,
                            Immediate_Unfilled=day_short,Inventory_Position=position))
    holding=area/PARAMS['operating_days_year']*PARAMS['holding_cost_unit_year']
    summary=dict(Initial_Stock=initial,End_Net_Inventory=net,Orders=orders,
                 Receipts=receipts,Immediate_Unfilled=total_short,
                 Fill_Rate_pct=100*(1-total_short/np.sum(demand)),
                 Days_With_Unfilled=int(sum(r['Immediate_Unfilled']>1e-8 for r in history)),
                 Completed_Cycles=len(completed),Successful_Cycles=int(sum(completed)),
                 Cycle_Service_pct=100*np.mean(completed) if completed else None,
                 Ordering_Cost=orders*PARAMS['order_cost'],Holding_Cost=holding,
                 Observed_Relevant_Cost=orders*PARAMS['order_cost']+holding)
    return pd.DataFrame(history),summary,pd.DataFrame(events)


def diagnostics(y,days):
    beta,X,target,keep=regression_fit(y,days)
    residual=target-X@beta;n,p=X.shape;df=n-p
    inverse=np.linalg.pinv(X.T@X)
    leverage=np.sum((X@inverse)*X,axis=1)
    adjusted=residual/np.maximum(1-leverage,1e-10)
    covariance=inverse@(X.T@((adjusted**2)[:,None]*X))@inverse
    se=np.sqrt(np.maximum(np.diag(covariance),0))
    stat=np.divide(beta,se,out=np.zeros_like(beta),where=se>0)
    coef=pd.DataFrame(dict(Term=['Intercept', *(WEEKDAY_NAMES[d] for d in DAY_CODES[:-1]), 'Lag1'],
                           Estimate=beta,HC3_SE=se,t_stat=stat,
                           p_value=2*student_t.sf(np.abs(stat),df)))
    sse=float(residual@residual);sst=float(np.sum((target-target.mean())**2))
    r2=1-sse/sst
    center=residual-residual.mean();denom=center@center
    acf=np.array([float(center[h:]@center[:-h]/denom) for h in range(1,6)])
    lb=float(n*(n+2)*np.sum(acf**2/(n-np.arange(1,6))))
    # Auxiliary regression of squared residuals for an approximate BP diagnostic.
    sq=residual**2;fit_sq=X@np.linalg.lstsq(X,sq,rcond=None)[0]
    r2sq=1-np.sum((sq-fit_sq)**2)/np.sum((sq-sq.mean())**2)
    return coef,dict(N=n,DF=df,R2=r2,Adjusted_R2=1-(1-r2)*(n-1)/df,
                     Residual_Mean=float(residual.mean()),Residual_SD=float(np.sqrt(sse/df)),
                     LB5=lb,LB5_p=float(chi2.sf(lb,5)),BP=float(n*r2sq),
                     BP_p=float(chi2.sf(n*r2sq,p-1)),ACF=acf.tolist()),residual,X@beta


def make_figures(data,future,test,policy,calibration,residual,fitted,plan,folder,winner):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(name,title,xlabel='Observation number',ylabel='Order-load units'):
        ax=plt.gca();ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.18)
        plt.tight_layout();plt.savefig(folder/name,dpi=220);plt.close()
    plt.figure(figsize=(9,4));plt.plot(data.Observation,data.Total_Demand)
    for bound,label in [(24.5,'Validation'),(40.5,'Test')]:plt.axvline(bound,ls='--',label=label)
    plt.legend();save('01_orders.png','Training validation and final test records')
    plt.figure(figsize=(9,4));plt.plot(policy.Step,policy.Net_Inventory,label='Net stock')
    plt.axhline(plan['reorder_point'],ls='--',label='Fixed reorder point')
    plt.axhline(0,color='black',lw=.7);plt.legend()
    save('02_policy.png','Continuous review (s,Q) scenario on twenty test days','Test operating day','Equivalent units')
    for model in MODELS:
        plt.figure(figsize=(9,4));plt.plot(np.arange(46,61),data.Total_Demand.iloc[-15:],label='Recent actuals')
        plt.plot(np.arange(61,66),future[model].iloc[:5],'o--',label=model);plt.legend()
        save('03_'+model+'.png',model+' five-step forecast')
    plt.figure(figsize=(9,4));plt.plot(test.Observation,test.Actual,color='black',label='Actual')
    for model in MODELS:plt.plot(test.Observation,test[model],label=model,alpha=.8)
    plt.legend(ncol=2);save('04_evaluation.png','Final test rolling one-step predictions')
    plt.figure(figsize=(9,4))
    for model in MODELS:plt.plot(future.Forecast_Step,future[model],label=model)
    plt.legend(ncol=2);save('05_future.png','Sixty-step forecasts after full-history refit','Forecast step')
    plt.figure(figsize=(9,4));means=data.groupby('Day_of_week').Total_Demand.mean()
    plt.bar(range(2,7),means.reindex(range(2,7)));plt.xticks(range(2,7),['Mon','Tue','Wed','Thu','Fri'])
    save('06_weekday.png','Mean observed demand by weekday','Recorded weekday')
    fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].scatter(fitted,residual);axes[0].axhline(0,c='black',lw=.7)
    axes[0].set(xlabel='Fitted demand',ylabel='Residual',title='Development residuals')
    center=residual-residual.mean();acf=[center[h:]@center[:-h]/(center@center) for h in range(1,6)]
    axes[1].bar(range(1,6),acf);axes[1].axhline(0,c='black',lw=.7)
    for sign in [-1,1]:axes[1].axhline(sign*1.96/np.sqrt(len(residual)),ls='--',color='gray')
    axes[1].set(xlabel='Retained-record lag',ylabel='Autocorrelation',title='Residual ACF')
    plt.tight_layout();plt.savefig(folder/'07_diagnostics.png',dpi=220);plt.close()
    plt.figure(figsize=(9,4));plt.bar(calibration.Origin,calibration.Error)
    plt.axhline(calibration.Error.mean(),ls='--',label='Mean cumulative error');plt.legend()
    save('08_lead_errors.png','Development five-step cumulative forecast errors','Forecast origin','Actual minus forecast total')
    plt.figure(figsize=(9,4));q=plan['EOQ'];grid=np.linspace(q*.3,q*2,100)
    plt.plot(grid,plan['annual_demand']/grid*50+grid/2*2+plan['safety_stock']*2)
    plt.axvline(q,ls='--',label='EOQ minimum');plt.legend()
    save('09_cost.png','Annual ordering and holding cost approximation','Equivalent order quantity','Currency units per year')
    fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
    labels=['Validate data','Tune on training','Select on validation','Calibrate lead errors','Evaluate final test','Refit and plan']
    for i,label in enumerate(labels):
        x=.02+i*.165
        ax.text(x+.065,.5,label.replace(' ','\n'),ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#edf1f4'))
        if i<5:ax.annotate('',xy=(x+.155,.5),xytext=(x+.135,.5),arrowprops=dict(arrowstyle='->'))
    plt.tight_layout();plt.savefig(folder/'10_flow.png',dpi=220);plt.close()


def main():
    tables,figures=ROOT/'outputs/tables',ROOT/'outputs/figures'
    for folder in [tables,figures,ROOT/'data/cleaned']:folder.mkdir(parents=True,exist_ok=True)
    data=load_data();y=data.Total_Demand.to_numpy();days=data.Day_of_week.to_numpy(int)
    data.to_csv(ROOT/'data/cleaned/cleaned_orders.csv',index=False)
    data[NAMES].describe().to_csv(tables/'descriptive_statistics.csv')
    gaps=[]
    for i in range(1,60):
        if not consecutive(days[i-1],days[i]):
            gaps.append(dict(Previous_Observation=i,Current_Observation=i+1,
                             Previous_Code=int(days[i-1]),Current_Code=int(days[i]),
                             Minimum_Skipped_Weekdays=((int(days[i])-int(days[i-1]))%5 or 5)-1))
    pd.DataFrame(gaps).to_csv(tables/'weekday_gaps.csv',index=False)
    window,tuning=choose_window(y[:24],days[:24]);tuning.to_csv(tables/'sma_tuning.csv',index=False)
    winner,validation=choose_model(y[:40],days[:40],window)
    validation.to_csv(tables/'validation_metrics.csv',index=False)
    predictions=pd.DataFrame(dict(Observation=np.arange(41,61),Actual=y[40:]))
    rows=[];origin_rows=[]
    for model in MODELS:
        rolling=np.array([predict(model,y[:i],days[:i],[days[i]],window)[0] for i in range(40,60)])
        fixed=predict(model,y[:40],days[:40],days[40:],window)
        predictions[model]=rolling;predictions[model+'_fixed20']=fixed
        rows.extend([dict(Model=model,Protocol='Rolling one step',**score(y[40:],rolling)),
                     dict(Model=model,Protocol='Fixed 20 step',**score(y[40:],fixed))])
        multi=[]
        for origin in [40,45,50,55]:
            fc=predict(model,y[:origin],days[:origin],days[origin:origin+5],window)
            multi.extend(fc)
            origin_rows.append(dict(Model=model,Origin=origin,**score(y[origin:origin+5],fc)))
        rows.append(dict(Model=model,Protocol='Four five-step origins',**score(y[40:],multi)))
    metrics=pd.DataFrame(rows);metrics.to_csv(tables/'model_metrics.csv',index=False)
    predictions.to_csv(tables/'test_predictions.csv',index=False)
    pd.DataFrame(origin_rows).to_csv(tables/'five_step_origins.csv',index=False)
    calibration=lead_calibration(y[:40],days[:40],winner,window)
    calibration.to_csv(tables/'lead_time_errors.csv',index=False)
    future_days=((int(days[-1])-2+np.arange(1,61))%5)+2
    future=pd.DataFrame(dict(Forecast_Step=np.arange(1,61),Weekday_Code=future_days))
    for model in MODELS:future[model]=predict(model,y,days,future_days,window)
    future.to_csv(tables/'forecast_60_steps.csv',index=False);future.head(5).to_csv(tables/'forecast_5_steps.csv',index=False)
    development_future_days=((int(days[39])-2+np.arange(1,61))%5)+2
    development_forecast=predict(winner,y[:40],days[:40],development_future_days,window)
    frozen=inventory(development_forecast.mean(),development_forecast[:5].sum(),calibration.Error.to_numpy())
    final=inventory(future[winner].mean(),future[winner].iloc[:5].sum(),calibration.Error.to_numpy())
    pd.DataFrame([dict(Basis='Frozen before final test',**frozen),dict(Basis='Full history planning',**final)]).to_csv(tables/'inventory_metrics.csv',index=False)
    policy,policy_summary,events=simulate_policy(y[40:],frozen['EOQ'],frozen['reorder_point'],5)
    policy.to_csv(tables/'policy_simulation.csv',index=False);events.to_csv(tables/'policy_events.csv',index=False)
    _,stress_summary,_=simulate_policy(y[40:]*1.3,frozen['EOQ'],frozen['reorder_point'],5)
    pd.DataFrame([dict(Scenario='Recorded test demand',**policy_summary),dict(Scenario='Demand multiplied by 1.30',**stress_summary)]).to_csv(tables/'policy_summary.csv',index=False)
    coef,diag,residual,fitted=diagnostics(y[:40],days[:40]);coef.to_csv(tables/'regression_coefficients.csv',index=False)
    (tables/'regression_diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
    sensitivity=[]
    for level in [.90,.95,.99]:sensitivity.append(dict(Service_Target=level,**inventory(future[winner].mean(),future[winner].iloc[:5].sum(),calibration.Error.to_numpy(),level)))
    pd.DataFrame(sensitivity).to_csv(tables/'service_sensitivity.csv',index=False)
    make_figures(data,future,predictions,policy,calibration,residual,fitted,frozen,figures,winner)
    summary=dict(parameters=PARAMS,selected_model=winner,sma_window=window,
                 ses_training=fit_ses(y[:24]),ses_development=fit_ses(y[:40]),ses_full=fit_ses(y),
                 means=dict(training=float(y[:24].mean()),validation=float(y[24:40].mean()),test=float(y[40:].mean()),all=float(y.mean())),
                 gaps=gaps,lead_calibration_N=len(calibration),lead_calibration_origins=calibration.Origin.tolist(),
                 frozen_inventory=frozen,final_inventory=final,policy=policy_summary,stress_policy=stress_summary,
                 historical_std=float(y.std(ddof=1)),development_diagnostics=diag,
                 final_DR_coefficients=regression_fit(y,days)[0].tolist(),
                 raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
                 versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    (tables/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text='VALIDATION METRICS\n'+validation.to_string(index=False)+'\n\nFINAL TEST METRICS\n'+metrics.to_string(index=False)
    text+='\n\nMODEL SELECTED BEFORE TEST: '+winner+'\n\nFINAL PLANNING SCENARIO\n'+json.dumps(final,indent=2)
    text+='\n\nPOLICY SIMULATION\n'+json.dumps(policy_summary,indent=2)+'\n'
    (ROOT/'outputs/run_log.txt').write_text(text,encoding='utf-8');print(text)


if __name__=='__main__':
    main()

