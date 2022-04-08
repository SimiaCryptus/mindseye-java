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
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * This class represents the standard deviation meta layer.
 *
 * @author John Doe
 * @version 1.0
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class StdDevMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StdDevMetaLayer.class);

  /**
   * Instantiates a new Std dev meta layer.
   */
  public StdDevMetaLayer() {
    this(1);
  }

  /**
   * Instantiates a new Std dev meta layer.
   *
   * @param minBatchCount the min batch count
   */
  public StdDevMetaLayer(final int minBatchCount) {
    super(1);
    AvgMetaLayer avgMetaLayer = new AvgMetaLayer();
    avgMetaLayer.setMinBatchCount(minBatchCount);
    RefUtil.freeRef(add(avgMetaLayer.addRef()));
    RefUtil.freeRef(add(new AvgReducerLayer()));
    InnerNode square = add(new SqActivationLayer());
    RefUtil.freeRef(add(new SqActivationLayer(), getInput(0)));
    RefUtil.freeRef(add(avgMetaLayer));
    RefUtil.freeRef(add(new AvgReducerLayer()));
    LinearActivationLayer negative = new LinearActivationLayer();
    negative.setScale(-1);
    negative.freeze();
    RefUtil.freeRef(add(new SumInputsLayer(), getHead(), add(negative, square)));
    NthPowerActivationLayer sqrt = new NthPowerActivationLayer();
    sqrt.setPower(0.5);
    RefUtil.freeRef(add(sqrt));
  }

  /**
   * Instantiates a new Std dev meta layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected StdDevMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * Creates a new StdDevMetaLayer from the given JSON object.
   *
   * @param json The JSON object to use.
   * @param rs   The map of resources to use.
   * @return The new StdDevMetaLayer.
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static StdDevMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StdDevMetaLayer(json, rs);
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  StdDevMetaLayer addRef() {
    return (StdDevMetaLayer) super.addRef();
  }

}
