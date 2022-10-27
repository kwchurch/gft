# Recommended Hyper-Parameter Settings

<table>
<tr>   <th>Dataset</th>         <th>Batch Size            </th>            <th>Learning Rate</th> </tr>
<tr>   <td>P:glue,cola</td>     <td> 32/64(base)          </td>            <td> 3e-5             </td> </tr>
<tr>   <td>P:glue,sst-2</td>    <td> 64/256(base)         </td>            <td> 2e-5            </td> </tr>
<tr>   <td>P:glue,sts-b</td>    <td> 128                  </td>            <td> 5e-5            </td> </tr>
<tr>   <td>P:glue,qqp</td>      <td> 256                  </td>            <td> 3e-5(base)/5e-5 (large)            </td> </tr>
<tr>   <td>P:glue,mnli</td>     <td> 256/512(base)        </td>            <td> 3e-5            </td> </tr>
<tr>   <td>P:glue,qnli</td>     <td> 256                  </td>            <td> 2e-5            </td> </tr>
<tr>   <td>P:glue,rte</td>      <td> 16/4(base)           </td>            <td> 2e-5 (base)/3e-5 (large) </td> </tr>
<tr>   <td>P:glue,mrpc</td>     <td> 16/32(base)          </td>            <td> 3e-5            </td> </tr>
<tr>   <td>P:glue,wnli</td>     <td> 8                    </td>            <td> 2e-5            </td> </tr>
<tr>   <td>P:glue,xnli</td>     <td> 512                  </td>            <td> 1e-4(base)/4e-5(large)            </td> </tr>
<tr>   <td>P:cmrc2018</td>      <td> 64                   </td>            <td> 3e-5            </td> </tr>
<tr>   <td>P:drcd</td>          <td> 64                   </td>            <td> 5e-4(base)/3e-5(large) </td> </tr>
<tr>   <td>P:msra-ner</td>      <td> 16                   </td>            <td> 5e-4(base)/1e-5(large) </td> </tr>
<tr>   <td>P:chnsenticorp</td>  <td> 24                   </td>            <td> 5e-4(base)/1e-5(large) </td> </tr>
<tr>   <td>P:lcqmc</td>         <td> 32                   </td>            <td> 2e-4(base)/5e-5(large) </td> </tr>
<tr>   <td>P:nlpcc2016-dbqa</td><td>64                    </td>            <td> 2e-4(base)/1e-5(large) </td> </tr>
<tr>   <td>P:vcr</td>           <td> 64                   </td>            <td> 2e-4(base)/2e-5(large) </td> </tr>
</table>
