//
// Created by rsltgy on 18/12/2020.
//

#include "Reader.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <sstream>

using namespace std;

void Reader::read_word_vector(string folder, unordered_map<string,vector<double>> &word_embeddings){
  ifstream glove_file;
  glove_file.open(folder);
  if(!glove_file){
    cout << "unable to word vector file" << '\n';
    exit(0);
  }
  cout << "Reading Word Embeddings" << endl;
  string word;
  string str;
  while(getline( glove_file,str)){
    istringstream ss(str);
    ss >> word;
    vector<double> embed;
    double elem;
    for(int s = 0 ; s < 25 ; s++) {
      ss >> elem;
      embed.push_back(elem);
    }
    word_embeddings[word] = embed;
  }
  //std::cout << "word embedding size " << word_embeddings.size() << endl;
}



void Reader::calculate_word_vector(unordered_map<string,vector<double>> &word_embeddings,string word,
                                   vector<double> &v_g_word_vector)
{

  vector<string> each_word;
  boost::split(each_word, word, boost::is_any_of(" "));
  unsigned int cnt = 0;
  for(string token : each_word){
    if(!token.empty()){
      if(word_embeddings.find(token) != word_embeddings.end()){
        cnt++;
        vector<double> each_word_vector = word_embeddings[token];
        if(!v_g_word_vector.empty())
          std::transform (v_g_word_vector.begin(), v_g_word_vector.end(),
                          each_word_vector.begin(), v_g_word_vector.begin(),
                          std::plus<double>());
        else
          v_g_word_vector = each_word_vector;
      }
    }
  }
  return;
  /*if(v_g_word_vector.empty()) return;
  cnt = each_word.size();
  transform(v_g_word_vector.begin(), v_g_word_vector.end(), v_g_word_vector.begin(), [cnt](double &c){ return c/cnt; });*/

  //if (v_g_word_vector.empty()) return;
  /*if (each_word.size() != cnt){
    vector<double> rand_vector (word_embeddings.begin()->second.size(),0);
    srand(time(0));
    generate(rand_vector.begin(), rand_vector.end(), rand);
    std::transform(rand_vector.begin(), rand_vector.end(), rand_vector.begin(),
                   [](double d) -> double { return d / RAND_MAX; });

    std::transform (v_g_word_vector.begin(), v_g_word_vector.end(),
                    rand_vector.begin(), v_g_word_vector.begin(),
                    std::plus<double>());

    int size = each_word.size();
    transform(v_g_word_vector.begin(), v_g_word_vector.end(), v_g_word_vector.begin(), [size](double &c){ return c/size; });
  }else{
    transform(v_g_word_vector.begin(), v_g_word_vector.end(), v_g_word_vector.begin(), [cnt](double &c){ return c/cnt; });
  }*/
}