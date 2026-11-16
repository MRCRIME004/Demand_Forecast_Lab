"""Presentation only: reads completed analysis values, never mutates them."""
import io,base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

TYPES={'01_demand':'line','02_evaluation':'line with bands','03_future':'line with bands','04_SES':'line with markers','04_SMA':'line with markers','04_Weekday_DR':'line with markers','04_Weekday_Naive':'line with markers','05_weekday':'box plot','06_errors':'dumbbell','07_residuals':'scatter','08_acf':'stem','09_lead_errors':'lollipop','10_policy':'stepped line and backlog area','11_cost':'stacked area and cost line','12_service':'stacked horizontal bars','13_coverage':'dot plot','14_cv_folds':'forest plot','15_dm_test':'diverging lollipop','16_feature_importance':'ranked horizontal bars','17_stl_decomposition':'stacked lines','18_stationarity':'lines and rolling spread','19_distribution':'histogram and box plot','20_demand_acf_pacf':'stem plots'}

def correlation_advice(values,ci,labels):
    significant=[[i+1 for i,v in enumerate(vs) if abs(v)>ci] for vs in values]
    finding='; '.join(f'{label} significant lags: {v or "none"}' for label,v in zip(labels,significant))
    advice='No lag exceeds the reference band; do not infer strong serial dependence from this chart.' if not any(significant) else 'Inspect the listed lags and compare forecast performance; correlation does not establish causation.'
    return finding,advice

def redraw(c,files,figures,include_supplementary=True):
    frames=c['frames']; s=c['summary']; winner=s['selected_model']; labels=s['model_labels']; colors=c['MODEL_COLORS']; u=s['unit_name']; name=labels[winner]
    y=np.asarray(c['y']); inp=frames['input_data']; test=frames['test_predictions']; future=frames['forecast']; n=len(y)
    day_codes=tuple(c.get('DAY_CODES',range(2,7)));day_labels=[c['WEEKDAY_NAMES'][d][:3] for d in day_codes];day_count=len(day_codes)
    day_term='calendar days' if day_count==7 else 'operating days'
    plt.rcParams.update({'font.size':13,'axes.titlesize':15,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':110})
    def fig(rows=1,cols=1):
        f,a=plt.subplots(rows,cols,figsize=(12,6.5),squeeze=False,sharex=rows==4,layout='constrained');return f,a.ravel()
    def style(a,x='',unit=True):
        a.set_xlabel(x);a.set_ylabel(u if unit else '');a.grid(alpha=.16);a.set_axisbelow(True);a.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    def legend(f):
        hs=[];ls=[]
        for a in f.axes:
            h,l=a.get_legend_handles_labels()
            for hi,li in zip(h,l):
                if li not in ls:hs.append(hi);ls.append(li)
        if hs:f.legend(hs,ls,loc='outside lower center',ncol=min(4,len(ls)),frameon=False)
    def encode(f,title):
        f.suptitle(title if len(title)<=80 else title[:77]+'…',fontsize=15,fontweight='bold');legend(f)
        b=io.BytesIO();f.savefig(b,format='png',dpi=110);plt.close(f);raw=b.getvalue();return raw,'data:image/png;base64,'+base64.b64encode(raw).decode()
    def guide(shows,read,understood,next_step):return dict(shows=shows,how_to_read=read,understood=understood,next_step=next_step)
    def bands(a,d,x,m):
        for level,alpha in [(95,.12),(80,.25)]:
            lo=f'{m}_lower_{level}';hi=f'{m}_upper_{level}'
            if lo in d:a.fill_between(x,d[lo].to_numpy(float),d[hi].to_numpy(float),color=colors[m],alpha=alpha,label=f'{level}% interval')
    def forecast_plot(key,m):
        f,aa=fig();a=aa[0]
        if m=='Comparison':
            d=test if key=='02_evaluation' else frames['forecast_5_steps'] if key.startswith('04_') else future
            x=d.Observation.to_numpy() if key=='02_evaluation' else n+d.Forecast_Step.to_numpy()
            if key=='02_evaluation':a.plot(x,d.Actual,color='#334155',lw=2,label='Actual')
            for i,model in enumerate(s['models_evaluated']):a.plot(x,d[model],color=colors[model],marker=['o','s','^','D','v','P','X','<','>','*'][i%10],ls=['-','--',':','-.'][i%4],ms=3,lw=1,label=labels[model])
            title='Forecast comparison across evaluated models';style(a,'Operating observation')
            return f,title,guide('Forecast lines for all evaluated models.','Match each line to its model label; compare against actual observations where available.','Model selection uses validation MAE; this view does not change the selected model.','Use the individual model view to inspect its interval bands.')
        if key=='02_evaluation':
            x=test['Observation'].to_numpy();a.plot(x,test.Actual,color='#334155',marker='o',ms=3,label='Actual');a.plot(x,test[m],color=colors[m],marker='s',ms=3,label=labels[m]);bands(a,test,x,m)
            title=f'Final test: actual demand and {labels[m]}'
            metric=frames['model_metrics'];metric=metric[(metric.Model==m)&(metric.Protocol=='Rolling one step')]
            text=f'Final test MAE for {labels[m]} is {float(metric.MAE.iloc[0]):,.2f} {u}.'
        else:
            d=frames['forecast_5_steps'] if key.startswith('04_') else future;tail=15 if key.startswith('04_') else 20
            a.plot(np.arange(n-tail+1,n+1),y[-tail:],color='#64748b',label='Observed');x=n+d.Forecast_Step.to_numpy();a.plot(x,d[m],color=colors[m],marker='o',ms=3,label=labels[m]);bands(a,d,x,m);a.axvline(n+.5,color='#475569',ls='--',label='Forecast begins')
            title=f'Next {len(d)} {day_term}: {labels[m]}'
            supported=int(np.isfinite(d[f'{m}_lower_80']).sum());text=f'Total forecast {float(d[m].sum()):,.2f} {u}; interval estimates are available for {supported} of {len(d)} steps.'
        if key=='02_evaluation':
            advice='This is the validation-selected model; use its held-out test error when planning tolerances.' if m==winner else f'This alternative view does not change the selected model, {name}; compare its validation and test errors before using it.'
        else:
            advice='Bands cover only part of this horizon; later unbanded steps do not have enough calibration errors for an interval.' if supported<len(d) else 'Bands are available for every displayed step; allow for uncertainty rather than treating the line as a guaranteed outcome.'
        style(a,'Operating observation');return f,title,guide('Observed and forecast lines with 80% and 95% interval bands.','Compare lines at the same operating observation; wider bands show more uncertainty. Missing bands mean insufficient calibration errors.',text,advice)
    def draw(key,m=winner):
        if key in ['02_evaluation','03_future'] or key.startswith('04_'):return forecast_plot(key,m)
        f,aa=fig(4 if key=='17_stl_decomposition' else 2 if key in ['10_policy','18_stationarity','20_demand_acf_pacf'] else 1,2 if key=='16_feature_importance' else 1);a=aa[0];title='';read='Read positions against the labelled axes; the guide states the source and scope.';under='';next_step='Use this diagnostic alongside the validation and final test tables.'
        if key=='01_demand':
            x=np.arange(1,n+1);a.plot(x,y,color='#334155',lw=1.5,label='Observed');ends=[0,s['split']['train_end'] if 'train_end' in s['split'] else c['train'],c['dev'],n]
            for lo,hi,col,label in zip(ends[:-1],ends[1:],['#e2e8f0','#fef3c7','#dcfce7'],['Training','Validation','Final test']):a.axvspan(lo+.5,hi+.5,color=col,label=label)
            for d in ends[1:-1]:a.axvline(d+.5,color='#64748b',ls=':')
            a.axhline(y.mean(),color='#475569',ls='--',label='Overall mean');flags=frames['stl_decomposition'].Outlier_Flag.to_numpy(bool);a.scatter(x[flags],y[flags],marker='x',color='#b91c1c',s=65,label='Flagged outlier');title='Observed demand across training, validation and test';under=f'{n:,} observations; mean {y.mean():,.2f} {u}; {flags.sum()} flagged outliers.';style(a,'Observation')
        elif key=='05_weekday':
            groups=[y[inp.Day_of_week.to_numpy()==d] for d in day_codes]
            if min(map(len,groups))>=8:
                a.boxplot(groups,tick_labels=day_labels,showmeans=True,meanprops=dict(marker='D',markerfacecolor='#2563eb',markeredgecolor='#1e3a8a'));under='Boxes show the middle half, a median line and diamond mean; whiskers and points show spread.'
            else:
                for i,v in enumerate(groups,1):
                    if len(v):a.scatter(i+np.linspace(-.12,.12,len(v)),v,color='#64748b',alpha=.6);a.vlines(i,min(v),max(v),color='#94a3b8');a.scatter(i,np.median(v),marker='_',s=200,color='#172033');a.scatter(i,np.mean(v),marker='D',color='#2563eb')
                a.set_xticks(range(1,day_count+1),day_labels);under='Dots show observations; horizontal marks show medians and diamonds show means.'
            title='Demand spread by weekday';style(a,'Weekday')
        elif key=='06_errors':
            d=frames['model_leaderboard'].sort_values('Validation_MAE');d=d[d.Model==m] if m!='Comparison' else d;pos=np.arange(len(d));a.hlines(pos,d.Validation_MAE,d.Test_MAE,color='#94a3b8')
            for i,r in enumerate(d.itertuples()):a.scatter(r.Validation_MAE,i,marker='o',color=colors[r.Model]);a.scatter(r.Test_MAE,i,marker='s',facecolors='none',edgecolors=colors[r.Model])
            a.scatter([],[],marker='o',color='#475569',label='Validation');a.scatter([],[],marker='s',facecolors='none',edgecolors='#475569',label='Final test');a.set_yticks(pos,[label+(' (winner)' if model==winner else '') for label,model in zip(d.Model_Label,d.Model)]);a.invert_yaxis();a.set_xlabel(f'MAE ({u})');title='Model comparison: validation vs final test MAE';read='Filled circle = validation; outlined square = final test. Shorter error is better.';under=f'{name} was selected by validation MAE.'
        elif key=='07_residuals':
            x=np.asarray(c['fitted']);r=np.asarray(c['residual']);a.scatter(x,r,color=colors['Weekday_DR'],alpha=.6);order=np.argsort(x);a.plot(x[order],pd.Series(r[order]).rolling(max(3,len(r)//8),center=True,min_periods=1).mean(),color='#b45309',label='Rolling mean');a.axhline(0,color='#334155',ls='--');title='Development regression residuals versus fitted demand';style(a,f'Fitted demand ({u})');a.set_ylabel(f'Actual − fitted ({u})');under=f'{len(r)} development regression residuals, rather than final test errors.'
        elif key in ['08_acf','20_demand_acf_pacf']:
            if key=='08_acf':vals=[np.asarray(c['diag']['ACF'])];ci=1.96/np.sqrt(len(c['residual']));labs=['Residual ACF']
            else:d=frames['demand_acf_pacf'];vals=[d.ACF.to_numpy(),d.PACF.to_numpy()];ci=float(d.CI_95_Upper.iloc[0]);labs=['Demand ACF','Demand PACF']
            for ax,v,lab in zip(aa,vals,labs):
                x=np.arange(1,len(v)+1);marker,stems,base=ax.stem(x,v,basefmt='k-');tone=colors['Weekday_DR'] if key=='08_acf' else '#334155';marker.set_color(tone);stems.set_color(tone);ax.axhspan(-ci,ci,color='#dbeafe',label='95% reference band');ax.set_title(lab);ax.set_xlabel('Lag (operating days)');ax.set_ylabel('Correlation');ax.grid(alpha=.15)
            title='Residual autocorrelation' if key=='08_acf' else 'Demand ACF and PACF: significant lags versus reference band';under,next_step=correlation_advice(vals,ci,labs)
        elif key=='09_lead_errors':
            if m=='Comparison':
                mm=sorted(c['lead_means'],key=c['lead_means'].get);a.barh([labels[k] for k in mm],[c['lead_means'][k] for k in mm],color=[colors[k] for k in mm]);a.set_xlabel(f'Mean absolute cumulative error ({u})');title='Lead-time error comparison by model';under='Errors use the existing development calibration windows.'
            else:
                d=c['lead_plot_values'][m];a.vlines(d.Origin,0,d.Error,color=colors[m]);a.scatter(d.Origin,d.Error,color=colors[m]);a.axhline(0,color='#334155');a.axhline(d.Error.mean(),ls='--',color='#b45309',label='Mean error');title=f'Cumulative lead-time error: {labels[m]}';style(a,'Forecast origin');under=f'{len(d)} windows; mean actual − forecast error {d.Error.mean():,.2f} {u}.'
        elif key=='10_policy':
            p=c['purchase_plan'];d=pd.DataFrame(p['forward_simulation_rows']) if p else frames['policy_simulation'];q=p['forward_simulation_summary'] if p else s['policy'];p=p or {**s['frozen_inventory'],'current_stock':q['Initial_Stock']};x=np.arange(1,len(d)+1);a.step(x,d.On_Hand,where='post',color='#0f766e',label='On-hand stock');a.axhline(p['reorder_point'],ls='--',color='#b45309',label='Reorder point')
            for col,marker,color,label in [('Ordered','v','#2563eb','Order placed'),('Received','^','#7c3aed','Receipt')]:mask=d[col]>0;a.scatter(x[mask],d.On_Hand[mask],marker=marker,color=color,s=60,label=label)
            aa[1].fill_between(x,d.Backlog,step='post',color='#b91c1c',alpha=.25);aa[1].set_ylabel(f'Backlog ({u})');aa[1].set_xlabel('Forward operating day' if c['purchase_plan'] else 'Test operating day');a.set_ylabel(f'Stock ({u})');basis='Forward plan' if c['purchase_plan'] else 'Historical test';title=f'{basis}: {q["Fill_Rate_pct"]:.1f}% filled · {q["Orders"]} orders · {q["Receipts"]} receipts · {q["Days_With_Unfilled"]} shortage days';under=f'Starting stock {p["current_stock"]:,.2f} {u}; {q["Days_With_Unfilled"]} days have unfilled demand.';next_step='Review the backlog panel and schedule before executing purchases.'
        elif key=='11_cost':
            p=c['purchase_plan'] or {**s['final_inventory'],'order_cost':c['options']['order_cost'],'holding_cost_unit_year':c['options']['holding_cost_unit_year']};Q=p['EOQ'];annual=c['summary']['final_inventory']['annual_demand'];xs=np.linspace(.4*Q,2*Q,200);ordering=annual*p['order_cost']/xs;cycle=xs*p['holding_cost_unit_year']/2;buffer=np.full_like(xs,p['annual_buffer_holding']);a.stackplot(xs,ordering,cycle,buffer,labels=['Ordering','Cycle holding','Buffer holding'],colors=['#bfdbfe','#99f6e4','#fde68a']);a.plot(xs,ordering+cycle+buffer,color='#0f172a',label='Total cost');a.scatter([Q],[p['annual_relevant_cost']],color='#0f172a');a.annotate(f'EOQ {Q:,.2f}',(Q,p['annual_relevant_cost']),xytext=(12,25),textcoords='offset points');title='Annual cost around the economic order quantity';a.set_xlabel(f'Order batch ({u})');a.set_ylabel('Annual relevant cost');under=f'EOQ {Q:,.2f} {u}; annual relevant cost {p["annual_relevant_cost"]:,.2f}.'
        elif key=='12_service':
            d=frames['service_sensitivity'];a.barh(d.Service_Label,d.reorder_point-d.safety_stock,label='Lead-time demand',color='#94a3b8');a.barh(d.Service_Label,d.safety_stock,left=d.reorder_point-d.safety_stock,label='Safety stock',color='#2563eb',hatch='//');title='Reorder point: lead-time demand plus safety buffer';a.set_xlabel(f'Reorder point ({u})');under='Safety stock changes with service option; EOQ is independent of the service target.'
        elif key=='13_coverage':
            d=frames['interval_coverage'];pos=np.arange(len(d));cc=[colors[m] for m in d.Model];a.scatter(d.Empirical_80_Coverage_pct,pos,marker='o',c=cc);a.scatter(d.Empirical_95_Coverage_pct,pos,marker='s',facecolors='none',edgecolors=cc);a.scatter([],[],marker='o',color='#475569',label='80% band');a.scatter([],[],marker='s',facecolors='none',edgecolors='#475569',label='95% band');a.axvline(80,ls='--',color='#64748b');a.axvline(95,ls=':',color='#64748b');a.set_yticks(pos,d.Model_Label);a.set_xlabel('Observed final-test coverage (%)');a.set_xlim(0,102);title='Observed interval coverage against nominal targets';under='Dots show observed test coverage; vertical lines mark nominal 80% and 95% targets.'
        elif key=='14_cv_folds':
            d=frames['cv_summary'].sort_values('Mean_MAE');fold=frames['cv_fold_metrics']
            if m!='Comparison':d=d[d.Model==m]
            for i,r in enumerate(d.itertuples()):a.errorbar(r.Mean_MAE,i,xerr=r.SD_MAE,fmt='o',color=colors[r.Model],capsize=4);a.scatter(fold.loc[fold.Model==r.Model,'MAE'],np.full(sum(fold.Model==r.Model),i),marker='.',color=colors[r.Model],alpha=.6)
            a.set_yticks(np.arange(len(d)),d.Model_Label);a.invert_yaxis();a.set_xlabel(f'CV MAE ({u}); dot = mean, line = ±1 SD');title='Cross-validation mean error and variability by model';under=f'{d.iloc[0].Model_Label} has the lowest mean CV MAE ({d.iloc[0].Mean_MAE:,.2f} {u}).'
        elif key=='15_dm_test':
            d=frames['diebold_mariano'];d=d[(d.Benchmark=='Weekday_Naive')&(d.Loss=='Absolute error (MAE)')];d=d.drop_duplicates('Model');v=d.DM_HLN_Stat.to_numpy();pos=np.arange(len(d));cc=['#0f766e' if p<.05 and val<0 else '#b91c1c' if p<.05 and val>0 else '#b45309' for p,val in zip(d.p_value,v)];a.hlines(pos,0,v,color=cc)
            for i,(val,p,color) in enumerate(zip(v,d.p_value,cc)):a.scatter(val,i,color=color,marker='D' if p<.05 and val<0 else 's' if p<.05 and val>0 else 'o')
            for marker,color,label in [('D','#0f766e','Significantly better'),('s','#b91c1c','Significantly worse'),('o','#b45309','Not significant')]:a.scatter([],[],marker=marker,color=color,label=label)
            a.axvline(0,color='#334155');from scipy.stats import t
            crit=float(t.ppf(.975,max(1,len(test)-1)));a.axvline(-crit,ls='--',color='#64748b',label='5% two-sided reference');a.axvline(crit,ls='--',color='#64748b');a.set_yticks(pos,d.Model_Label);a.set_xlabel('HLN-adjusted DM statistic (negative favours candidate)');title='Diebold–Mariano comparison against weekday naive';under='Diamond positions show test statistics; significance requires crossing the reference threshold.'
        elif key=='16_feature_importance':
            d=frames['feature_importance'];clean=lambda v:str(v).replace('num__','').replace('cat__','').replace('_',' ')
            for ax,col,lab,color in [(aa[0],'Ridge_Std_Coef','Ridge standardized coefficient',colors['Weekday_Ridge']),(aa[1],'RF_Permutation_MAE_Increase','RF permutation ΔMAE',colors['Weekday_RF'])]:
                if col not in d:ax.text(.5,.5,'Model unavailable',ha='center',transform=ax.transAxes);continue
                z=d.sort_values(col);ax.barh([clean(v) for v in z.Feature],z[col],color=color);ax.axvline(0,color='#475569');ax.set_title(lab);ax.set_xlabel(u)
            title='Predictive feature importance: Ridge and Random Forest';under='Signed coefficients and permutation error changes measure predictive association, not causation.'
        elif key=='17_stl_decomposition':
            d=frames['stl_decomposition'];cols=['Treated_Demand','STL_Trend','STL_Seasonal','STL_Residual']
            for ax,col in zip(aa,cols):ax.plot(d.Observation,d[col],color='#334155');ax.set_ylabel(col.replace('STL_','').replace('_',' '),fontsize=11);ax.grid(alpha=.15)
            flags=d.Outlier_Flag.to_numpy(bool);aa[0].scatter(d.Observation[flags],d.Treated_Demand[flags],marker='x',color='#b91c1c');aa[-1].set_xlabel('Observation');title='STL decomposition: observed, trend, seasonal and residual';under=f'{flags.sum()} causal outlier flags are marked on the observed series.'
        elif key=='18_stationarity':
            for ax,v,label in [(aa[0],y,'Level'),(aa[1],np.diff(y),'First difference')]:
                ser=pd.Series(v);mean=ser.rolling(5,min_periods=1).mean();sd=ser.rolling(5,min_periods=2).std().fillna(0);x=np.arange(1,len(v)+1);ax.plot(x,v,color='#64748b',label='Series');ax.plot(x,mean,color='#2563eb',label='Rolling mean (5)');ax.fill_between(x,mean-sd,mean+sd,color='#2563eb',alpha=.12,label='Rolling ±1 SD');ax.set_title(label);ax.set_ylabel(u)
            st=frames['stationarity_tests']
            for i,ax in enumerate(aa):
                rows=st.iloc[i*2:i*2+2];text=' · '.join(f'{str(r.Test).split()[0]} p={r.p_value:.3f}' for r in rows.itertuples());verdict=str(rows.Combined_Verdict.iloc[0]).split(':')[0];ax.set_title(f'{"Level" if i==0 else "First difference"}: {verdict} · {text}',fontsize=11)
            title='Level and first difference: rolling mean and spread';under='; '.join(st.Combined_Verdict.drop_duplicates());aa[-1].set_xlabel('Observation')
        elif key=='19_distribution':
            plt.close(f);f=plt.figure(figsize=(12,6.5),layout='constrained');aa=f.subplots(2,1,gridspec_kw={'height_ratios':[1,4]});aa[0].boxplot(y,orientation='horizontal');aa[0].set_yticks([]);a=aa[1];a.hist(y,bins=c['dist_stats']['bins'],color='#94a3b8',edgecolor='white');a.axvline(y.mean(),color='#2563eb',ls='--',label='Mean');a.axvline(np.median(y),color='#0f766e',ls=':',label='Median');a.set_ylabel('Operating days');a.set_xlabel(u);title=f'Demand distribution: skewness {c["dist_stats"]["skewness"]:+.3f}' if 'skewness' in c['dist_stats'] else 'Demand distribution, median and spread';under=f'Mean {y.mean():,.2f} {u}; median {np.median(y):,.2f} {u}.'
        # Guide wording only: inspect completed results without changing frames,
        # model choices, metrics, inventory calculations or exported values.
        if key=='01_demand':next_step='Review flagged observations in Data before choosing an outlier treatment; a flag alone is not proof of an error.' if flags.any() else 'No observations are flagged; inspect the split regions and validate the forecasts before changing data treatment.'
        elif key=='05_weekday':
            ex=s['executive_summary'];under=f'{ex["peak_weekday"]} has the highest observed weekday mean ({ex["peak_weekday_mean"]:,.2f} {u}); {ex["low_weekday"]} has the lowest ({ex["low_weekday_mean"]:,.2f} {u}).'
            next_step='Weekday means differ; check spread and the validation errors before relying on that pattern.' if ex['peak_weekday_mean']>ex['low_weekday_mean'] else 'Weekday means are equal here; do not infer a weekday advantage from their averages.'
        elif key=='06_errors':next_step='The selected model’s final-test MAE is below its validation MAE; retain both estimates when discussing performance.' if s['test_metrics']['MAE']<s['selected_validation_MAE'] else 'The selected model’s final-test MAE is at least its validation MAE; allow for that held-out error in planning.'
        elif key=='07_residuals':next_step='Residuals occur on both sides of zero; inspect changes in spread and the rolling mean rather than assuming the fit is unbiased.' if np.any(r<0) and np.any(r>0) else 'Residuals do not occur on both sides of zero; inspect systematic bias before relying on the regression fit.'
        elif key=='09_lead_errors':
            if m=='Comparison':next_step=f'{labels[min(c["lead_means"],key=c["lead_means"].get)]} has the smallest mean absolute lead-time error; compare this development diagnostic with validation results.'
            else:next_step='Mean cumulative error is positive, so actual demand exceeded forecast on average in these windows; consider this when reviewing the buffer.' if d.Error.mean()>0 else 'Mean cumulative error is non-positive in these windows; inspect individual misses before changing the buffer.'
        elif key=='10_policy':next_step='This plan has unfilled demand; review the backlog panel and receipt timing before executing purchases.' if q['Days_With_Unfilled']>0 else 'This scenario has no unfilled-demand days; still review receipt timing and the model’s held-out errors before purchasing.'
        elif key=='11_cost':next_step='Buffer holding contributes to annual cost; verify the service target and cost inputs before choosing a practical batch.' if p['annual_buffer_holding']>0 else 'No buffer holding cost is included; verify that omitting a safety buffer suits the planning assumptions.'
        elif key=='12_service':next_step='Some service options include safety stock; compare their buffer costs without treating the target as an observed service guarantee.' if (d.safety_stock>0).any() else 'These service options add no safety stock; review the calibration and service inputs before relying on that result.'
        elif key=='13_coverage':
            row=d[d.Model==winner].iloc[0];under=f'{name}: observed 80% interval coverage {row.Empirical_80_Coverage_pct:,.2f}%; observed 95% coverage {row.Empirical_95_Coverage_pct:,.2f}%.'
            if not np.isfinite([row.Empirical_80_Coverage_pct,row.Empirical_95_Coverage_pct]).all():under=f'{name}: at least one empirical interval coverage estimate is unavailable.';next_step='Review calibration support before judging interval coverage; an unavailable rate does not meet a nominal target.'
            else:next_step='At least one selected-model coverage rate is below its nominal level; do not treat nominal intervals as an operational guarantee.' if row.Empirical_80_Coverage_pct<80 or row.Empirical_95_Coverage_pct<95 else 'Both selected-model coverage rates meet their nominal levels on this test sample; check the sample size before generalizing.'
        elif key=='14_cv_folds':
            if m!='Comparison':under=f'{d.iloc[0].Model_Label}: mean CV MAE {d.iloc[0].Mean_MAE:,.2f} {u}; SD {d.iloc[0].SD_MAE:,.2f} {u}.'
            cv_best=frames['cv_summary'].sort_values('Mean_MAE').iloc[0]
            next_step='The CV leader agrees with validation selection; compare fold spread and held-out test errors before using the forecast.' if cv_best.Model==winner else f'The CV leader ({cv_best.Model_Label}) differs from validation selection ({name}); investigate the difference without changing the selection rule.'
        elif key=='15_dm_test':
            row=d[d.Model==winner]
            if len(row):
                row=row.iloc[0];under=f'{name}: HLN-adjusted statistic {row.DM_HLN_Stat:,.3f}; p-value {row.p_value:.3f}.'
                if not np.isfinite(row.p_value):under=f'{name}: significance is unavailable for this comparison.';next_step='Read the test availability notes; an unavailable p-value cannot establish a difference.'
                else:next_step='The selected model differs significantly from weekday naive at 5%; read the statistic’s sign to determine which has lower loss.' if row.p_value<.05 else 'The selected model is not significantly different from weekday naive at 5%; do not claim proven superiority from this test.'
        elif key=='16_feature_importance':next_step='Both model importance summaries are available; compare associations and signs without interpreting them as causal effects.' if 'RF_Permutation_MAE_Increase' in d and 'Ridge_Std_Coef' in d else 'One importance summary is unavailable; read the available panel and runtime notes before drawing a comparison.'
        elif key=='17_stl_decomposition':next_step='Outlier flags are present; inspect those observations and the residual panel before choosing treatment.' if flags.any() else 'No outlier flags are present; compare trend and seasonal panels with validation performance before choosing a model.'
        elif key=='18_stationarity':next_step='The reported level verdict is stationary; compare it with the differenced verdict before adding differencing.' if str(st.Combined_Verdict.iloc[0]).startswith('Stationary:') else 'The level verdict is not a clear stationary result; inspect both ADF and KPSS results and the differenced panel before modeling.'
        elif key=='19_distribution':next_step='Skewness is positive; compare the mean and median and inspect high-demand observations before using a symmetric approximation.' if c['dist_stats'].get('skewness',0)>0 else 'Skewness is non-positive; compare the mean, median and tails before using a symmetric approximation.'
        descriptions={
          '01_demand':('Every observed demand value, with shaded chronological data splits.','Follow the line in time; dashed dividers separate training, validation and final test. Crosses mark flagged observations.'),
          '05_weekday':('Demand distributions grouped by operating weekday.','Compare medians, means and spread. An isolated point is an observation, not a forecast.'),
          '07_residuals':('Development regression errors against their fitted demand values.','Look for a curve or changing spread around zero; the orange line is a rolling residual mean.'),
          '08_acf':('Development regression residual correlation at successive lags.','Stems outside the shaded 95% reference band merit investigation.'),
          '09_lead_errors':('Cumulative development forecast errors over supplier lead-time windows.','Above zero means actual demand exceeded the forecast; the dashed line is the mean error.'),
          '10_policy':('Stock, order and receipt events, and backlog for the entered forward plan.' if c['purchase_plan'] else 'Historical test-stock simulation and backlog.','Triangles mark orders and receipts; the lower shaded area shows backlog, distinct from on-hand stock.'),
          '11_cost':('Annual ordering, cycle-holding and buffer-holding costs across batch sizes.','The dark total-cost line is the sum of the stacked components; the marked point is EOQ.'),
          '12_service':('Reorder point decomposed into lead-time demand and safety stock.','Read total horizontal length as ROP; the hatched segment is the safety buffer.'),
          '13_coverage':('Final-test coverage of 80% and 95% prediction intervals by model.','Circles and squares identify the interval level; compare their x-position with the corresponding nominal line.'),
          '14_cv_folds':('Model mean CV MAE, one-standard-deviation spans, and individual fold errors.','Smaller MAE is better; shorter spans indicate less fold-to-fold variation.'),
          '15_dm_test':('HLN-corrected comparison of model errors with weekday naive.','Negative statistics favour the candidate; crossing a significance threshold supplies stronger evidence.'),
          '16_feature_importance':('Ridge coefficients and Random Forest validation permutation importance.','Coefficients have direction; permutation values measure the change in MAE when a predictor is shuffled.'),
          '17_stl_decomposition':('The observed series and its STL trend, seasonal and residual components.','Read panels vertically at the same observation; component values use the original demand scale.'),
          '18_stationarity':('Demand level and its first difference with a five-observation rolling mean and spread.','The shaded range is rolling mean ±1 SD, not a forecast interval. Use the ADF/KPSS results for the statistical verdict.'),
          '19_distribution':('A histogram of historical demand with mean, median and a compact box plot.','Histogram heights are operating-day counts; the box plot summarizes median and spread.'),
          '20_demand_acf_pacf':('Demand ACF and PACF correlations across operating-day lags.','Read stems against the shaded 95% reference band; a stem within it is not flagged as significant.')}
        if key in descriptions:shows,read=descriptions[key]
        else:shows=f'{TYPES.get(key,"Chart").capitalize()} of completed results.'
        return f,title,guide(shows,read,under,next_step)
    for item in figures:
        key=item['key'];m=key[3:] if key.startswith('04_') else 'Comparison' if key in ['06_errors','14_cv_folds'] else winner
        f,title,g=draw(key,m);raw,url=encode(f,title);files[item['path']]=raw
        kind='strip plot with median, range and mean' if key=='05_weekday' and inp.Day_of_week.value_counts().reindex(day_codes,fill_value=0).min()<8 else TYPES[key]
        item.update(title=title,data=url,interpretation=g,chart_type=kind);item.pop('method_views',None)
        if key in ['02_evaluation','03_future','06_errors','09_lead_errors','14_cv_folds'] or key.startswith('04_'):
            views={}
            for model in s['models_evaluated']+['Comparison']:
                f,t,g=draw(key,model);_,urlv=encode(f,t);views[model]=dict(title=t,data=urlv,method=model,label=labels.get(model,'Comparison'),accent_color=colors.get(model,'#475569'),interpretation=g)
            item.update(method_views=views,default_method=m)
    if not include_supplementary:return figures
    def add(key,title,kind,group,frame,f,g):
        frames[key]=frame
        raw,url=encode(f,title);path=f'figures/{key}.png';files[path]=raw
        figures.append(dict(key=key,title=title,group=group,category=group,part=1,path=path,data=url,chart_type=kind,accent_color='#2563eb',interpretation=g))
    def matrix(a,values,rowlabels,collabels,label,count_text=None):
        v=np.asarray(values,float);im=a.imshow(np.ma.masked_invalid(v),cmap='Blues',aspect='equal',vmin=0,vmax=max(1,float(np.nanmax(v))) if np.isfinite(v).any() else 1)
        a.set_xticks(np.arange(len(collabels)),collabels);a.set_yticks(np.arange(len(rowlabels)),rowlabels);a.tick_params(axis='x',rotation=25)
        a.figure.colorbar(im,ax=a,label=label,shrink=.8)
        if max(v.shape)<=12:
            for i,j in np.ndindex(v.shape):
                if not np.isfinite(v[i,j]):continue
                rgb=im.cmap(im.norm(v[i,j]))[:3];lin=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in rgb];lum=sum(x*w for x,w in zip(lin,[.2126,.7152,.0722]));col='black' if (lum+.05)/.05>=1.05/(lum+.05) else 'white'
                a.text(j,i,count_text[i][j] if count_text else f'{v[i,j]:,.1f}',ha='center',va='center',color=col,fontsize=11)
    descriptive=' This chart is descriptive and does not affect any forecast.'
    cv=frames['cv_fold_metrics'];order=frames['cv_summary'].sort_values('Mean_MAE').Model.tolist();grid=cv.pivot(index='Model',columns='Fold',values='MAE').reindex(order);f,aa=fig();matrix(aa[0],grid.to_numpy(),[labels[x] for x in order],[f'Fold {x}' for x in grid.columns],f'MAE ({u})');title='Cross-validation error by model and fold'
    add('21_cv_heatmap',title,'annotated heatmap','Evaluation',cv[['Model','Model_Label','Fold','MAE']].copy(),f,guide('Each cell contains a model’s error in one CV fold.'+descriptive,'Lighter cells mean smaller MAE. Read across a row for stability.',f'{len(order)} models across {grid.shape[1]} folds.','Compare individual folds with the mean and SD in the forest plot.'))
    codes=inp.Day_of_week.to_numpy();week=np.r_[0,np.cumsum(codes[1:]<=codes[:-1])]+1;cal=pd.DataFrame({'Operating_Week':week,'Weekday_Code':codes,'Demand':y});grid=cal.pivot_table(index='Operating_Week',columns='Weekday_Code',values='Demand',aggfunc='mean').reindex(columns=day_codes).tail(26);f,aa=fig();matrix(aa[0],grid.to_numpy(),[str(x) for x in grid.index],day_labels,u);aa[0].set_ylabel('Operating week (inferred from weekday sequence)')
    add('22_demand_calendar_heatmap','Demand by operating week and weekday','heatmap','Exploration',cal,f,guide('Demand in up to 26 recent operating weeks; missing cells are blank. The CSV retains all weeks.'+descriptive,'Compare weekdays across a row and weeks down a column. Large grids omit annotations to remain readable.','Week groups follow weekday wraparound; weekday codes alone cannot reveal full missing weeks.','Use the chronological series to investigate unusually high or low cells.'))
    actual=test.Actual.to_numpy();pred=test[winner].to_numpy();mae=float(s['test_metrics']['MAE']);lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max());f,aa=fig();a=aa[0];a.fill_between([lo,hi],[lo-mae,hi-mae],[lo+mae,hi+mae],color=colors[winner],alpha=.12,label='±1 test MAE');a.plot([lo,hi],[lo,hi],color='#334155',ls='--',label='Perfect forecast');a.scatter(actual,pred,color=colors[winner],marker='o');a.set_xlabel(f'Actual ({u})');a.set_ylabel(f'Forecast ({u})');a.set_aspect('equal',adjustable='box')
    add('23_actual_vs_forecast',f'Actual versus forecast: {name}','scatter','Evaluation',pd.DataFrame({'Observation':test.Observation,'Actual':actual,'Forecast':pred,'Absolute_Error':np.abs(pred-actual)}),f,guide('Each dot pairs actual and predicted final-test demand.'+descriptive,'Above the diagonal = over-forecast; below = under-forecast. The shaded strip is ±1 MAE, not a confidence interval.',f'Test MAE is {mae:,.2f} {u}.','Inspect departures from the diagonal; use the time-series chart to locate these days.'))
    if len(test)>=20:
        weekdays=inp.set_index('Observation').loc[test.Observation,'Day_of_week'].to_numpy();direction=np.where(pred-actual>actual*.05,'Over-forecast',np.where(pred-actual< -actual*.05,'Under-forecast','Within ±5%'));columns=['Over-forecast','Within ±5%','Under-forecast'];counts=pd.crosstab(weekdays,direction).reindex(index=day_codes,columns=columns,fill_value=0);tot=counts.sum(axis=1).to_numpy();pct=counts.to_numpy()/np.maximum(tot[:,None],1)*100;txt=[[f'{counts.iloc[i,j]}\n({pct[i,j]:.1f}%)' for j in range(3)] for i in range(day_count)];f,aa=fig();matrix(aa[0],pct,day_labels,columns,'Share within weekday (%)',txt);long=pd.DataFrame([dict(Weekday=d,Direction=col,Count=int(counts.loc[d,col]),Share_pct=float(pct[i,j])) for i,d in enumerate(counts.index) for j,col in enumerate(columns)])
        add('24_error_direction_matrix','Final-test error direction by weekday','annotated matrix','Diagnostics',long,f,guide('Counts and percentages of over, near, and under forecasts by weekday.'+descriptive,'Percentages use each weekday’s test-row total; a zero-count weekday is shown as 0%.',f'{len(test)} final-test observations; near means within ±5% of actual demand.','Check whether error direction clusters by weekday before changing operational decisions.'))
    return figures
