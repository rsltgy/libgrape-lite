//
// Created by rsltgy on 20/12/2020.
//

#ifndef LIBGRAPE_LITE_APAIR_H
#define LIBGRAPE_LITE_APAIR_H


#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Apair_context.h"
#include "grape/utils/vertex_array.h"
#include "../KDTree/KDTree.h"

using namespace std;


namespace grape{

template <typename FRAG_T>
class Apair : public ParallelAppBase<FRAG_T, ApairContext<FRAG_T>>,
public ParallelEngine {


// specialize the templated worker.
INSTALL_PARALLEL_WORKER(Apair<FRAG_T>, ApairContext<FRAG_T>, FRAG_T)
using vertex_t = typename fragment_t::vertex_t;
void PEval(const fragment_t& frag, context_t& ctx,
           message_manager_t& messages) {

  auto &rev = ctx.rev;
  auto &match_set = ctx.match_set;
  auto &C = ctx.C;
  auto &GD = ctx.GD;
  auto &cache = ctx.cache;
  auto &ecache_u = ctx.ecache_u;
  auto &ecache_v = ctx.ecache_v;
  auto &g_paths = ctx.g_paths;
  auto &g_descendants = ctx.g_descendants;
  auto &word_embeddings = ctx.word_embeddings;
  auto &sigma = ctx.sigma;
  auto &delta = ctx.delta;
  messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
  auto& channel_0 = messages.Channels()[0];

  pointVec points;
  auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    string v_str = frag.GetData(v);
    //auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
    //cout << frag.fid() << " " << oid  << " " << v_str << endl;
    vector<double> v_g_word_vector;
    boost::trim_right(v_str);
    Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);
    if(!v_g_word_vector.empty()){
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      point_t t = make_pair(oid,v_g_word_vector);
      points.push_back(t);
      //cout << v_str <<  " " << oid << endl;

    }
  }

  KDTree tree(points);

  for(int u_t = 0; u_t < GD.number_of_nodes(); u_t++){
    //cout << frag.fid() << " " <<  u_t << endl;
    if(!GD.nodes()[u_t].empty()){
      vector<double> u_t_word_vector;
      Reader::calculate_word_vector(word_embeddings,GD.nodes()[u_t],u_t_word_vector);
      if(!u_t_word_vector.empty()) {
        point_t t = make_pair(u_t,u_t_word_vector);
        auto NNs = tree.neighborhood_points(t, sigma);
        for(auto returned_match : NNs){
          //vertex_t v(returned_match.first);
          //auto es = frag.GetOutgoingAdjList(v);
          C.push_back(make_pair(u_t,make_pair(returned_match.first,5)));
          //cout << frag.fid() << " " <<  u_t << " " << returned_match.first << endl;
        }
      }
    }

  }
  cout << " candidates generated " << endl;
  // We do not need the tree anymore, clear out.
  points.clear();
  points.shrink_to_fit();


  sort(C.begin(),C.end(),[](const pair<int,pair<int,int>> &a, const pair<int,pair<int,int>> b){
    return a.second.second < b.second.second;
  });

  bool match = false;
  for(auto v : C){
    //cout << frag.fid() << " ) " << " " << v.first << " " << v.second.first << endl;
    auto it = cache.find(make_pair(v.first,v.second.first));
    if(it != ctx.cache.end() && cache[make_pair(v.first,v.second.first)].first  ){
      match_set.push_back(make_pair(v.first,v.second.first));
      //cout << " here " << v.first<< " " << v.second.first << endl;
    }
    else{
      ParaMatch<FRAG_T> p;
      match = p.match_pair(GD,frag,g_paths,g_descendants,v.first,v.second.first,sigma,delta,cache,word_embeddings,ecache_u,ecache_v,rev);
        if(!match){ // if spair of u and v is false, then do further calculation otherwise do nothing.
            ctx.v = v.second.first;
            ctx.u = v.first;
            vertex_t frag_vert;
            vertex_t o_v;
            if(frag.GetInnerVertex(v.second.first,frag_vert)){                // if vertex v is in this fragment then send message to other
                auto outer_vertices = frag.OuterVertices();
                map<unsigned int,std::pair<unsigned int,vector<std::pair<int,int>>>> msg;

                for (vertex_t o_v : outer_vertices ) {
                    unsigned int fid = frag.GetFragId(o_v);
                    msg[fid].first = frag.fid();
                    //for(int n = 0 ; n < GD.number_of_nodes(); n++)
                    for(int n = 0 ; n < 1; n++) // instead of all GD, just send a key to check all vertices in GD
                        msg[fid].second.push_back(std::make_pair(n,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
                }
                for (auto m : msg ) {
                    channel_0.SendToFragment(m.first,m.second);
                }

                auto witness_vertices =  cache[make_pair(ctx.u,v.second.first)].second;
                double &sum = ctx.sum;
                ParaMatch<FRAG_T> p;
                for(auto wit : witness_vertices){
                    double local_sum = p.calculate_path_similarity(GD,g_paths,ctx.word_embeddings, ctx.u, wit.first, v.second.first, wit.second);
                    sum += local_sum;
                    //cout << " sum of " << ctx.u << " " << wit.first << " " << v.second.first << " " << wit.second  << " is " << local_sum << " total " << sum << endl;
                }
            }
        }else{
            match_set.push_back(make_pair(v.first,v.second.first));
        }

    }

  }
    cout << " PEval Done" << endl;

}




void IncEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {

    auto &rev = ctx.rev;
    auto &GD = ctx.GD;
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
                        for(int n = 0 ; n < GD.number_of_nodes(); n++)
                            msg[fid].second.push_back(std::make_pair(n,frag.Gid2Oid(frag.Vertex2Gid(o_v))));
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
                                                                   y.first, ctx.v, y.second);
                    sum += local_sum;
                    /*cout << " sum of " << ctx.u << " " << y.first << " " << ctx.v << " " << y.second << " is "
                         << local_sum << " total " << sum << endl;*/
                    if (sum >= ctx.delta) {
                        //cout << sum << " Vertex u " << ctx.u << " and " << ctx.v << " is a match " << endl;
                        ctx.result = true;
                        match_set.push_back(make_pair(ctx.u,ctx.v));
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


#endif  // LIBGRAPE_LITE_APAIR_H
