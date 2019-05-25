/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.lang.ref.ReferenceCounting;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.CountingResult;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * This key works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger png regions.
 */
@SuppressWarnings("serial")
public class StochasticSamplingSubnetLayer extends LayerBase implements StochasticComponent {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StochasticSamplingSubnetLayer.class);

  private final int samples;
  @Nullable
  private final Layer subnetwork;
  private long seed = System.nanoTime();

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param subnetwork the subnetwork
   * @param samples    the samples
   */
  public StochasticSamplingSubnetLayer(final Layer subnetwork, final int samples) {
    super();
    this.samples = samples;
    this.subnetwork = subnetwork;
    this.subnetwork.addRef();
  }

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected StochasticSamplingSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    samples = json.getAsJsonPrimitive("samples").getAsInt();
    seed = json.getAsJsonPrimitive("seed").getAsLong();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : Layer.fromJson(subnetwork, rs);
  }

  /**
   * From json rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet key
   */
  public static StochasticSamplingSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StochasticSamplingSubnetLayer(json, rs);
  }

  /**
   * Average result.
   *
   * @param samples the samples
   * @return the result
   */
  public static Result average(final Result[] samples) {
    PipelineNetwork gateNetwork = new PipelineNetwork(samples.length);
    try {
      gateNetwork.wrap(new SumInputsLayer(), IntStream.range(0, samples.length).mapToObj(i -> gateNetwork.getInput(i)).toArray(i -> new DAGNode[i])).freeRef();
      gateNetwork.wrap(new LinearActivationLayer().setScale(1.0 / samples.length).freeze()).freeRef();
      return gateNetwork.evalAndFree(samples);
    } finally {
      gateNetwork.freeRef();
    }
  }

  @NotNull
  public static StochasticSamplingSubnetLayer wrap(Layer subnet, int samples) {
    StochasticSamplingSubnetLayer stochasticSamplingSubnetLayer = new StochasticSamplingSubnetLayer(subnet, samples);
    subnet.freeRef();
    return stochasticSamplingSubnetLayer;
  }

  @Override
  protected void _free() {
    this.subnetwork.freeRef();
    super._free();
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if(0 == seed) {
      return subnetwork.evalAndFree(inObj);
    }
    Result[] counting = Arrays.stream(inObj).map(r -> {
      return new CountingResult(r, samples);
    }).toArray(i -> new Result[i]);
    Arrays.stream(inObj).forEach(ReferenceCounting::freeRef);
    Result average = average(Arrays.stream(getSeeds()).mapToObj(seed -> {
      shuffleSubnet(seed);
      return subnetwork.eval(counting);
    }).toArray(i -> new Result[i]));
    Arrays.stream(counting).map(Result::getData).forEach(ReferenceCounting::freeRef);
    Arrays.stream(counting).forEach(ReferenceCounting::freeRef);
    return average;
  }

  public void shuffleSubnet(long seed) {
    if (subnetwork instanceof DAGNetwork) {
      ((DAGNetwork) subnetwork).visitNodes(node -> {
        Layer layer = node.getLayer();
        if (layer instanceof StochasticComponent) {
          ((StochasticComponent) layer).shuffle(seed);
        }
      });
    }
    if (subnetwork instanceof StochasticComponent) {
      ((StochasticComponent) subnetwork).shuffle(seed);
    }
  }

  /**
   * Get seeds long [ ].
   *
   * @return the long [ ]
   */
  public long[] getSeeds() {
    Random random = new Random(seed);
    return IntStream.range(0, this.samples).mapToLong(i -> random.nextLong()).toArray();
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("samples", samples);
    json.addProperty("seed", seed);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }


  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    subnetwork.setFrozen(frozen);
    return super.setFrozen(frozen);
  }

  @Override
  public void shuffle(final long seed) {
    log.info(String.format("Set %s to random seed %s", getName(), seed));
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    log.info(String.format("Set %s to random null seed", getName()));
    seed = 0;
  }

}
