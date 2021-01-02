//
// Created by rsltgy on 19/12/2020.
//

#ifndef LIBGRAPE_LITE_VPAIR_H
#define LIBGRAPE_LITE_VPAIR_H

#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Vpair_context.h"
#include "grape/utils/vertex_array.h"

using namespace std;


namespace grape{

template <typename FRAG_T>
class Vpair : public ParallelAppBase<FRAG_T, VpairContext<FRAG_T>>,
public ParallelEngine {


// specialize the templated worker.
INSTALL_PARALLEL_WORKER(Vpair<FRAG_T>, VpairContext<FRAG_T>, FRAG_T)
using vertex_t = typename fragment_t::vertex_t;


    //static constexpr MessageStrategy message_strategy =
    //        MessageStrategy::kAlongEdgeToOuterVertex;

    static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

void PEval(const fragment_t& frag, context_t& ctx,
           message_manager_t& messages) {

  /*auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
    std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
  }*/
  auto &rev = ctx.rev;
  auto &match_set = ctx.match_set;
  auto &C = ctx.C;
  auto &GD = ctx.GD;
  auto &u = ctx.u;
  auto &cache = ctx.cache;
  auto &ecache_u = ctx.ecache_u;
  auto &ecache_v = ctx.ecache_v;
  auto &g_paths = ctx.g_paths;
  auto &g_descendants = ctx.g_descendants;
  auto &word_embeddings = ctx.word_embeddings;
  double &sigma = ctx.sigma;
  double &delta = ctx.delta;
  messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
  auto& channel_0 = messages.Channels()[0];

  ParaMatch<FRAG_T> p;

  string u_label = GD.nodes()[u];
  vector<double> u_t_word_vector;
  Reader::calculate_word_vector(ctx.word_embeddings,u_label,u_t_word_vector);

  auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    //auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
    //std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
    string v_str = frag.GetData(v);
    //cout << gd.nodes()[u] << " " << v_str << endl;
    vector<double> v_g_word_vector;
    boost::trim_right(v_str);
    Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);

    double score = p.cosine_similarity(u_t_word_vector,v_g_word_vector);

    if(score >= sigma ) {
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      auto es = frag.GetOutgoingAdjList(v);
      //std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
      C.push_back(make_pair(oid, es.Size()));
    }

  }

  sort(C.begin(),C.end(),[](const pair<int,int> &a, const pair<int,int>b){
    return a.second < b.second;
  });

  bool match = false;
  for(auto c : C){
    //cout << frag.fid() << " --> " << u << " " << v.first << endl;
    auto it = ctx.cache.find(make_pair(u,c.first));
    if(it != ctx.cache.end()){
        if(ctx.cache[make_pair(u,c.first)].first){
          ctx.match_set.push_back(c.first);
        }
    }
    else{
      int v_vertex = c.first;
      match = p.match_pair(GD,frag,g_paths,g_descendants,u,v_vertex,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
        if(!match){ // if spair of u and v is false, then do further calculation otherwise do nothing.
            ctx.v = v_vertex;
            vertex_t frag_vert;
            vertex_t o_v;
            if(frag.GetInnerVertex(v_vertex,frag_vert)){                // if vertex v is in this fragment then send message to other
                auto outer_vertices = frag.OuterVertices();
                map<unsigned int,std::pair<unsigned int,vector<std::pair<int,int>>>> msg;

                for (vertex_t o_v : outer_vertices ) {
                    unsigned int fid = frag.GetFragId(o_v);
                    msg[fid].first = frag.fid();
                    //for(int n = 0 ; n < GD.number_of_nodes(); n++)
                    for(int n = 0 ; n < 1; n++) // instead of all GD, just send a key to check all vertices in GD
                        msg[fid].second.push_back(std::make_pair(-1,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
                }
                for (auto m : msg ) {
                    channel_0.SendToFragment(m.first,m.second);
                }

                auto witness_vertices =  cache[make_pair(ctx.u,v_vertex)].second;
                double &sum = ctx.sum;
                ParaMatch<FRAG_T> p;
                for(auto wit : witness_vertices){
                    double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, ctx.u, wit.first, v_vertex, wit.second);
                    sum += local_sum;
                    cout << " sum of " << ctx.u << " " << wit.first << " " << v_vertex << " " << wit.second  << " is " << local_sum << " total " << sum << endl;
                }
            }
        }else{
            match_set.push_back(v_vertex);
        }

    }

  }

}




    void IncEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {

        auto &rev = ctx.rev;
        auto &GD = ctx.GD;
        auto &v = ctx.v;
        auto &cache = ctx.cache;
        auto &ecache_u = ctx.ecache_u;
        auto &ecache_v = ctx.ecache_v;
        auto &g_paths = ctx.g_paths;
        auto &g_descendants = ctx.g_descendants;
        auto &word_embeddings = ctx.word_embeddings;
        double &sigma = ctx.sigma;
        double &delta = ctx.delta;
        auto& channel_0 = messages.Channels()[0];
        double &sum = ctx.sum;
        auto &match_set = ctx.match_set;
        ParaMatch<FRAG_T> p;

        vector<std::pair<unsigned int,vector<std::pair<int,int>>>>  messages_received;
        messages.ParallelProcess<std::pair<unsigned int,vector<std::pair<int,int>>>>(
                1, [&frag,&messages_received](int tid, std::pair<unsigned int,vector<std::pair<int,int>>> msg) {
                    messages_received.push_back(msg);
                });

        for(auto message : messages_received){
            if(!message.second.empty()) {
                bool fragment_has_everything = false;
                for (auto m : message.second) {
                    auto pair_received = std::make_pair(m.first, m.second);
                    if(m.first == -1){
                        // This call comes from Peval, execute all GD.
                        for(int n = 0 ; n < GD.number_of_nodes(); n++){
                            p.match_pair(GD, frag, g_paths, g_descendants, n, m.second, sigma, delta, cache,
                                         word_embeddings, ecache_u, ecache_v,rev);
                            if(cache[std::make_pair(n, m.second)].first){
                                fragment_has_everything = true;
                                break;
                            }
                        }
                    }else if(cache.find(pair_received) == cache.end()){
                        p.match_pair(GD, frag, g_paths, g_descendants, m.first, m.second, sigma, delta, cache,
                                     word_embeddings, ecache_u, ecache_v,rev);
                        fragment_has_everything = true;
                    }
                }
                if(fragment_has_everything){
                    map<unsigned int,std::pair<unsigned int,vector<std::pair<int,int>>>> msg;
                    for (auto &ca : cache) {
                        /*cout << "Fragment " << frag.fid() << " (" << ca.first.first
                             << "," << ca.first.second << ") -> " << ca.second.first << " " << ca.second.second.size()
                             << endl;*/
                        if (ca.second.first) {
                            vertex_t frag_vert;
                            frag.GetVertex(ca.first.first, frag_vert);
                            msg[message.first].first = frag.fid();
                            msg[message.first].second.push_back(ca.first);
                        }
                    }

                    if(msg.empty()){   // This is for cycles
                        for (vertex_t o_v : frag.OuterVertices()) {
                            unsigned int fid = frag.GetFragId(o_v);
                            msg[fid].first = frag.fid();
                            for(int n = 0 ; n < 1; n++)
                                msg[fid].second.push_back(std::make_pair(-1,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
                        }
                    }
                    for (auto m : msg ) {
                        //cout << " frag " << frag.fid() << " is sending a message to frag" << m.first << endl;
                        channel_0.SendToFragment(m.first,m.second);
                    }
                }else{
                    for (auto y : message.second) {
                        /*cout << " frag " << frag.fid() << " got message from " << message.first << " " << y.first
                             << " " << y.second << endl;*/
                        double local_sum = p.calculate_path_similarity(GD, g_paths, ctx.word_embeddings, ctx.u,
                                                                       y.first, v, y.second);
                        sum += local_sum;
                        /*cout << " sum of " << ctx.u << " " << y.first << " " << v << " " << y.second << " is "
                             << local_sum << " total " << sum << endl;*/
                        if (sum >= ctx.delta) {
                            cout << sum << " Vertex u " << ctx.u << " and " << ctx.v << " is a match " << endl;
                            ctx.result = true;
                            match_set.push_back(ctx.v);
                            ctx.sum = 0;
                            break;
                        }
                    }


                }
            }
        }

}

};



}


#endif  // LIBGRAPE_LITE_VPAIR_H
