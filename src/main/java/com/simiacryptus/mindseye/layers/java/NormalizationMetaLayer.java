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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * This class is responsible for the NormalizationMetaLayer.
 *
 * @author Author Name
 * @version 1.0
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class NormalizationMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizationMetaLayer.class);

  /**
   * Instantiates a new Normalization meta layer.
   */
  public NormalizationMetaLayer() {
    super(1);
    RefUtil.freeRef(add(new SqActivationLayer()));
    RefUtil.freeRef(add(new AvgReducerLayer()));
    RefUtil.freeRef(add(new AvgMetaLayer()));
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(-0.5);
    RefUtil.freeRef(add(nthPowerActivationLayer));
    RefUtil.freeRef(add(new ProductInputsLayer(), getHead(), getInput(0)));
  }

  /**
   * Instantiates a new Normalization meta layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected NormalizationMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * Creates a new NormalizationMetaLayer from the given JSON object.
   *
   * @param json The JSON object to use.
   * @param rs   A map of character sequences to byte arrays.
   * @return The new NormalizationMetaLayer.
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static NormalizationMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new NormalizationMetaLayer(json, rs);
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
  NormalizationMetaLayer addRef() {
    return (NormalizationMetaLayer) super.addRef();
  }

}
