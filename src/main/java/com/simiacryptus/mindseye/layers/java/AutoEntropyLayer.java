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
 * The AutoEntropyLayer class is a utility class that is used to
 * automatically calculate the entropy of a given data layer.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class AutoEntropyLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AutoEntropyLayer.class);

  /**
   * Instantiates a new Auto entropy layer.
   */
  public AutoEntropyLayer() {
    super(1);
    RefUtil.freeRef(add(new EntropyLossLayer(), getInput(0), getInput(0)));
  }

  /**
   * Instantiates a new Auto entropy layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected AutoEntropyLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * Creates an AutoEntropyLayer from a JSON object.
   *
   * @param json the JSON object to create the layer from
   * @param rs   the map of character sequences to byte arrays
   * @return the AutoEntropyLayer created from the JSON object
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static AutoEntropyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AutoEntropyLayer(json, rs);
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
  AutoEntropyLayer addRef() {
    return (AutoEntropyLayer) super.addRef();
  }

}
