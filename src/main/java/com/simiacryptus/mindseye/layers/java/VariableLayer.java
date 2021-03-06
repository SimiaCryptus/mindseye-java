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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * The type Variable layer.
 */
@SuppressWarnings("serial")
public class VariableLayer extends WrapperLayer {

  /**
   * Instantiates a new Variable layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected VariableLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * Instantiates a new Variable layer.
   *
   * @param inner the inner
   */
  public VariableLayer(final Layer inner) {
    super(inner);
  }

  @Override
  public RefList<Layer> getChildren() {
    return super.getChildren();
  }

  /**
   * From json variable layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the variable layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static VariableLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new VariableLayer(json, rs);
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  VariableLayer addRef() {
    return (VariableLayer) super.addRef();
  }

}
