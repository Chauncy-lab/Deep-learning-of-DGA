from sklearn.preprocessing import OneHotEncoder
import warnings
import numpy as np
warnings.filterwarnings("ignore")

#Batch大小
batch_size=200
#域名最大长度
max_domain_size=50
#生成的二级域名长度
g_domain_len=8
#鉴别器初始训练epochs
D_epochs=1
#DG训练的epochs
DG_epochs=1000
#一个DG epochs中生成器训练的steps
G_steps=7
#一个DG epochs中鉴别器训练的steps
D_steps=5
#鉴别器模型保存的位置
D_model_path="model/D_model"
#生成域名的根域名列表
roots=['.aaa', '.abb', '.abc', '.ac', '.aco', '.ad', '.ads', '.ae', '.aeg', '.af', '.afl', '.ag', '.ai', '.aig', '.al', '.am', '.an', '.anz', '.ao', '.aol', '.app', '.aq', '.ar', '.art', '.as', '.at', '.au', '.aw', '.aws', '.ax', '.axa', '.az', '.ba', '.bar', '.bb', '.bbc', '.bbt', '.bcg', '.bcn', '.bd', '.be', '.bet', '.bf', '.bg', '.bh', '.bi', '.bid', '.bio', '.biz', '.bj', '.bl', '.bm', '.bms', '.bmw', '.bn', '.bnl', '.bo', '.bom', '.boo', '.bot', '.box', '.bq', '.br', '.bs', '.bt', '.buy', '.bv', '.bw', '.by', '.bz', '.bzh', '.ca', '.cab', '.cal', '.cam', '.car', '.cat', '.cba', '.cbn', '.cbs', '.cc', '.cd', '.ceb', '.ceo', '.cf', '.cfa', '.cfd', '.cg', '.ch', '.ci', '.ck', '.cl', '.cm', '.cn', '.co', '.com', '.cr', '.crs', '.csc', '.cu', '.cv', '.cw', '.cx', '.cy', '.cz', '.dad', '.day', '.dds', '.de', '.dev', '.dhl', '.diy', '.dj', '.dk', '.dm', '.dnp', '.do', '.dog', '.dot', '.dtv', '.dvr', '.dz', '.eat', '.ec', '.eco', '.edu', '.ee', '.eg', '.eh', '.er', '.es', '.esq', '.et', '.eu', '.eus', '.fan', '.fi', '.fit', '.fj', '.fk', '.fly', '.fm', '.fo', '.foo', '.fox', '.fr', '.frl', '.ftr', '.fun', '.fyi', '.ga', '.gal', '.gap', '.gb', '.gd', '.gdn', '.ge', '.gea', '.gf', '.gg', '.gh', '.gi', '.gl', '.gle', '.gm', '.gmo', '.gmx', '.gn', '.goo', '.gop', '.got', '.gov', '.gp', '.gq', '.gr', '.gs', '.gt', '.gu', '.gw', '.gy', '.hbo', '.hiv', '.hk', '.hkt', '.hm', '.hn', '.hot', '.how', '.hr', '.ht', '.htc', '.hu', '.ibm', '.ice', '.icu', '.id', '.ie', '.ifm', '.il', '.im', '.in', '.ing', '.ink', '.int', '.io', '.iq', '.ir', '.is', '.ist', '.it', '.itv', '.iwc', '.jcb', '.jcp', '.je', '.jio', '.jlc', '.jll', '.jm', '.jmp', '.jnj', '.jo', '.jot', '.joy', '.jp', '.ke', '.kfh', '.kg', '.kh', '.ki', '.kia', '.kim', '.km', '.kn', '.kp', '.kpn', '.kr', '.krd', '.kw', '.ky', '.kz', '.la', '.lat', '.law', '.lb', '.lc', '.lds', '.li', '.lk', '.lol', '.lpl', '.lr', '.ls', '.lt', '.ltd', '.lu', '.lv', '.ly', '.ma', '.man', '.map', '.mba', '.mc', '.mcd', '.md', '.me', '.med', '.men', '.meo', '.mf', '.mg', '.mh', '.mil', '.mit', '.mk', '.ml', '.mlb', '.mls', '.mm', '.mma', '.mn', '.mo', '.moe', '.moi', '.mom', '.mov', '.mp', '.mq', '.mr', '.ms', '.msd', '.mt', '.mtn', '.mtr', '.mu', '.mv', '.mw', '.mx', '.my', '.mz', '.na', '.nab', '.nba', '.nc', '.ne', '.nec', '.net', '.new', '.nf', '.nfl', '.ng', '.ngo', '.nhk', '.ni', '.nl', '.no', '.now', '.np', '.nr', '.nra', '.nrw', '.ntt', '.nu', '.nyc', '.nz', '.obi', '.off', '.om', '.one', '.ong', '.onl', '.ooo', '.org', '.ott', '.ovh', '.pa', '.pay', '.pe', '.pet', '.pf', '.pg', '.ph', '.phd', '.pid', '.pin', '.pk', '.pl', '.pm', '.pn', '.pnc', '.pr', '.pro', '.pru', '.ps', '.pt', '.pub', '.pw', '.pwc', '.py', '.qa', '.qvc', '.re', '.red', '.ren', '.ril', '.rio', '.rip', '.ro', '.rs', '.ru', '.run', '.rw', '.rwe', '.sa', '.sap', '.sas', '.sb', '.sbi', '.sbs', '.sc', '.sca', '.scb', '.sd', '.se', '.ses', '.sew', '.sex', '.sfr', '.sg', '.sh', '.si', '.sj', '.sk', '.ski', '.sky', '.sl', '.sm', '.sn', '.so', '.soy', '.sr', '.srl', '.srt', '.ss', '.st', '.stc', '.su', '.sv', '.sx', '.sy', '.sz', '.tab', '.tax', '.tc', '.tci', '.td', '.tdk', '.tel', '.tf', '.tg', '.th', '.thd', '.tj', '.tjx', '.tk', '.tl', '.tm', '.tn', '.to', '.top', '.tp', '.tr', '.trv', '.tt', '.tui', '.tv', '.tvs', '.tw', '.tz', '.ua', '.ubs', '.ug', '.uk', '.um', '.uno', '.uol', '.ups', '.us', '.uy', '.uz', '.va', '.vc', '.ve', '.vet', '.vg', '.vi', '.vig', '.vin', '.vip', '.vn', '.vu', '.wed', '.wf', '.win', '.wme', '.wow', '.ws', '.wtc', '.wtf', '.xin', '.xxx', '.xyz', '.ye', '.you', '.yt', '.yun', '.za', '.zip', '.zm', '.zw']
#字符列表
chars=[".","-","_"]
alpha=[chr(i) for i in range(97,123)]
chars.extend(alpha)
num=[chr(i) for i in range(48,58)]
chars.extend(num)
chars_size=len(chars)

def to_onehot(s):
    s_index=[]
    for c in s:
        s_index.append(chars.index(c))
    one_hot=OneHotEncoder(n_values=chars_size)
    return one_hot.fit_transform(np.array(s_index).reshape(-1, 1)).toarray().reshape((len(s),chars_size))